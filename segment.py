import numpy as np
import torch
import pickle
from scene import Scene
import os
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import skip_feat_decoder
from scipy.spatial.transform import Rotation as R
import featsplat_editor
from einops import einsum
from typing import List

import open3d as o3d
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry

def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
    import open3d as o3d
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)

@torch.no_grad()
def select_gs_for_phys(dataset : ModelParams,
                  iteration : int,
                  fg_obj_list : List[str],
                  bj_obj_list : List[str],
                  ground_plane_name : str,
                  threshold : float,
                  object_select_eps : float,
                  inward_selection_eps : float,
                  final_noise_filtering : bool,
                  interactive_viz : bool,
                  rigid_object_name : str):
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    if interactive_viz:
        print("=====================================")
        print("Interactive visualization enabled")
        print("=====================================")

    ply_editing_dir = os.path.join(scene.model_path,
                                    "point_cloud",
                                    "iteration_" + str(scene.loaded_iter))

    my_feat_decoder = skip_feat_decoder(dataset.distill_feature_dim).cuda()
    decoder_weight_path = os.path.join(dataset.model_path, "feat_decoder.pth")
    assert os.path.exists(decoder_weight_path)
    decoder_weight_dict = torch.load(decoder_weight_path)
    my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
    my_feat_decoder.eval()

    clip_segmeter = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder, clip_device='cuda')
    ground_estimator = featsplat_editor.ground_estimator(rotation_flip=True)

    start_cp = time.time()
    fg_obj_bbox = clip_segmeter.fast_compute_rough_bbox(fg_obj_list)  # (2, 3)

    # Draw bbox
    if interactive_viz:
        print(bcolors.WARNING + "Check if the desired object is inside the bounding box" + bcolors.ENDC)
        input("Press enter to continue")
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(gaussians.get_xyz.cpu().numpy())

        center = (fg_obj_bbox[0] + fg_obj_bbox[1]) / 2
        size = fg_obj_bbox[1] - fg_obj_bbox[0]
        bbox = o3d.geometry.OrientedBoundingBox(center=center, R=np.eye(3), extent=size)

        o3d.visualization.draw_geometries([scene_pcd, bbox])

    # Create a subset of Gaussians
    bounded_xyz = gaussians.get_xyz
    within_bbox = ((bounded_xyz[:, 0] > fg_obj_bbox[0, 0]) & (bounded_xyz[:, 0] < fg_obj_bbox[1, 0])) & \
                    ((bounded_xyz[:, 1] > fg_obj_bbox[0, 1]) & (bounded_xyz[:, 1] < fg_obj_bbox[1, 1])) & \
                    ((bounded_xyz[:, 2] > fg_obj_bbox[0, 2]) & (bounded_xyz[:, 2] < fg_obj_bbox[1, 2]))
    bounded_xyz = bounded_xyz[within_bbox]
    bounded_xyz_np = bounded_xyz.cpu().numpy()
    current_idx = torch.arange(gaussians.get_xyz.shape[0])[within_bbox.cpu()]
    bounded_features = gaussians.get_distill_features[within_bbox]

    fg_obj_similarity = clip_segmeter.compute_similarity_one(fg_obj_list, feature=bounded_features)
    fg_obj_idx = fg_obj_similarity > threshold

    if interactive_viz:
        print(bcolors.WARNING + "Check if the desired object is selected (outliers will be removed)" + bcolors.ENDC)
        input("Press enter to continue")
        # Visualize selected points
        selected_pcd = o3d.geometry.PointCloud()
        selected_pcd.points = o3d.utility.Vector3dVector(bounded_xyz_np)
        selected_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * fg_obj_idx[:, None])
        o3d.visualization.draw_geometries([selected_pcd])
    
    selected_obj_idx = fg_obj_idx

    if interactive_viz:
        print(bcolors.WARNING + "Check if the clustered object is (roughly) correct" + bcolors.ENDC)
        input("Press enter to continue")
        while True:
            selected_obj_idx = clip_segmeter.cluster_instance(bounded_xyz_np,
                                                            fg_obj_idx,
                                                            eps=object_select_eps)
            # Visualize clustered points
            clustered_pcd = o3d.geometry.PointCloud()
            clustered_pcd.points = o3d.utility.Vector3dVector(bounded_xyz_np)
            clustered_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * selected_obj_idx[:, None])
            o3d.visualization.draw_geometries([clustered_pcd])
            result = input("If some particles are missing, increase eps. If too many noises, decrease eps. Current eps: {:.4f}; New eps: ".format(object_select_eps))
            if result == "":
                break
            else:
                provided_eps = float(result)
                if np.isclose(provided_eps, object_select_eps):
                    break
                object_select_eps = provided_eps
    else:
        selected_obj_idx = clip_segmeter.cluster_instance(bounded_xyz_np,
                                                        selected_obj_idx,
                                                        eps=object_select_eps)

    # Estimate ground plane using all Gaussians. Slower but more accurate
    ground_similarity = clip_segmeter.compute_similarity_one(ground_plane_name)
    ground_idx = ground_similarity > threshold
    ground_R, ground_T, ground_inliers = ground_estimator.estimate(gaussians.get_xyz.cpu().numpy()[ground_idx])
    # ground_idx, sampled_ground_idx = clip_segmeter.compute_similarity_on_downsampled(ground_plane_name)
    # ground_R, ground_T, ground_inliers = ground_estimator.estimate(gaussians.get_xyz[sampled_ground_idx][ground_idx].cpu().numpy())

    # A bounding box that selects the object with some noisy outer particles
    selected_obj_idx = clip_segmeter.ground_bbox_filter(bounded_xyz_np,
                                                        selected_obj_idx,
                                                        ground_R, ground_T,
                                                        boundary=np.array([0, 0, 0]))
    
    # Refine it
    bounded_xyz = bounded_xyz[selected_obj_idx]
    bounded_xyz_np = bounded_xyz.cpu().numpy()
    bounded_features = bounded_features[selected_obj_idx]
    current_idx = current_idx[selected_obj_idx]

    word_list = bj_obj_list + fg_obj_list
    text_features_mc = clip_segmeter.get_text_embeddings(word_list)

    chunk_feature_nc = clip_segmeter.decoder_infer(bounded_features, 'object')
    chunk_feature_nc = chunk_feature_nc / (chunk_feature_nc.norm(dim=1, keepdim=True) + 1e-6)
    similarity_nm = einsum(chunk_feature_nc.float(), text_features_mc.float(), 'n c, m c -> n m')

    positive_obj_idx = similarity_nm.argmax(dim=1) >= len(bj_obj_list)
    positive_obj_idx = positive_obj_idx.cpu().numpy()
    # fg_obj_sim = similarity_nm.softmax(dim=1)[:, -len(fg_obj_list):].sum(dim=1)

    # Capture particles inward
    if interactive_viz:
        print(bcolors.WARNING + "Selecting interior of the object. Check if it includes any noises" + bcolors.ENDC)
        input("Press enter to continue")
        obj_idx = positive_obj_idx
        while True:
            positive_obj_idx = clip_segmeter.ground_bbox_filter(bounded_xyz_np,
                                                                obj_idx,
                                                                ground_R, ground_T,
                                                                boundary=np.array([inward_selection_eps, inward_selection_eps, inward_selection_eps]))
            # Visualize clustered points
            clustered_pcd = o3d.geometry.PointCloud()
            clustered_pcd.points = o3d.utility.Vector3dVector(bounded_xyz_np)
            clustered_pcd.colors = o3d.utility.Vector3dVector(np.array([1, 0, 0]) * positive_obj_idx[:, None])
            o3d.visualization.draw_geometries([clustered_pcd])
            result = input("Increase inward bbox eps if it contain noises. Current eps: {:.4f}; New eps: ".format(inward_selection_eps))
            if result == "":
                break
            else:
                provided_eps = float(result)
                if np.isclose(provided_eps, inward_selection_eps):
                    break
                inward_selection_eps = provided_eps
    else:
        positive_obj_idx = clip_segmeter.ground_bbox_filter(bounded_xyz_np,
                                                            positive_obj_idx,
                                                            ground_R, ground_T,
                                                            boundary=np.array([inward_selection_eps, inward_selection_eps, inward_selection_eps]))
    
    # Get particles on the periphery of the bbox (that is not close to other surfaces)
    positive_obj_idx = clip_segmeter.knn_infilling(bounded_xyz_np,
                                            positive_obj_idx,
                                            dilation_iters=1,
                                            positive_ratio=0.5,
                                            k=20)
    
    # Adaptive ground filtering
    positive_obj_idx = clip_segmeter.remove_ground(bounded_xyz_np, positive_obj_idx, ground_R, ground_T)

    non_fg_obj_idx = ~positive_obj_idx
    non_fg_obj_idx = clip_segmeter.knn_infilling(bounded_xyz_np,
                                            non_fg_obj_idx,
                                            dilation_iters=1,
                                            positive_ratio=0.5,
                                            k=20)
    
    positive_obj_idx = ~non_fg_obj_idx

    # Final clustering; use 10% as minimum object distance
    if final_noise_filtering:
        guessed_eps = np.mean(bounded_xyz_np.max(axis=0) - bounded_xyz_np.min(axis=0)) / 10
        positive_obj_idx = clip_segmeter.cluster_instance(bounded_xyz_np, positive_obj_idx, eps=guessed_eps)

    print("Total segmentation time: ", time.time() - start_cp)

    up_gravity_vec = np.array((0, 1, 0))
    up_gravity_vec = ground_R.T @ up_gravity_vec

    rot_deg = 180
    rot_axis = up_gravity_vec / np.linalg.norm(up_gravity_vec)
    r = R.from_rotvec(rot_deg * rot_axis, degrees=True)
    rot_mat = r.as_matrix()

    translate_vec = np.array([1.0, 0, 1.7])
    translate_vec = ground_R.T @ translate_vec

    final_obj_flag = np.zeros(gaussians.get_xyz.shape[0], dtype=bool)
    final_obj_flag[current_idx] = positive_obj_idx

    if rigid_object_name:
        rigid_obj_similarity = clip_segmeter.compute_similarity_one(rigid_object_name, feature=gaussians.get_distill_features[final_obj_flag])
        rigid_obj_similarity = rigid_obj_similarity > 0.8  # keep only the most similar object
        # Filter out outliers
        non_rigid_obj_similarity = ~rigid_obj_similarity
        non_rigid_obj_idx = clip_segmeter.knn_infilling(gaussians.get_xyz.cpu().numpy()[final_obj_flag],
                                                        non_rigid_obj_similarity,
                                                        dilation_iters=1,
                                                        positive_ratio=0.4,
                                                        k=20)
        rigid_obj_similarity = ~non_rigid_obj_idx
        # Aggresive dilation
        rigid_obj_similarity = clip_segmeter.knn_infilling(gaussians.get_xyz.cpu().numpy()[final_obj_flag],
                                                        rigid_obj_similarity,
                                                        dilation_iters=3,
                                                        positive_ratio=0.2,
                                                        k=20)

    print(bcolors.WARNING + "Densifying surface Gaussians" + bcolors.ENDC)
    start_cp = time.time()
    binarized_voxel, center_xyz, scale_xyz = gaussians.extract_fields(mask=final_obj_flag, resolution=256,
                                                                        binarize_threshold=0.1)
    print("Voxelization time: ", time.time() - start_cp)
    
    # Visualize in open3d
    voxel_res = binarized_voxel.shape[0]
    assert voxel_res % 2 == 0

    voxel_viz_pcd = o3d.geometry.PointCloud()
    # Create 128^3 grid of points and select
    pts_on_disk_n3 = np.mgrid[0:voxel_res, 0:voxel_res, 0:voxel_res].reshape(3, -1).T
    pts_on_disk_n3 = pts_on_disk_n3[binarized_voxel.flatten() == 1]
    pts_on_disk_n3 = pts_on_disk_n3 / (voxel_res // 2) - 1
    pts_on_disk_n3 = (pts_on_disk_n3 / scale_xyz) + center_xyz
    print("Total infilling points: ", pts_on_disk_n3.shape[0])
    voxel_viz_pcd.points = o3d.utility.Vector3dVector(pts_on_disk_n3)

    if interactive_viz:
        print(bcolors.WARNING + "Final check: the selected object is good and x-z plane is aligned to ground" + bcolors.ENDC)
        input("Press enter to continue")
        arrow_mesh = get_arrow(up_gravity_vec)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

        # Rotate mesh frame to align with ground
        mesh_frame = mesh_frame.rotate(ground_R.T, center=(0, 0, 0))
        mesh_frame = mesh_frame.translate(-(ground_R.T @ ground_T))

        o3d_point_cloud = o3d.geometry.PointCloud()

        o3d_point_cloud.points = o3d.utility.Vector3dVector(bounded_xyz_np)

        color_arr = np.zeros((bounded_xyz_np.shape[0], 4))
        color_arr[positive_obj_idx, 0] = 1
        color_arr[positive_obj_idx, 3] = 1
        color_arr = color_arr[:, :3] * color_arr[:, 3:]

        o3d_point_cloud.colors = o3d.utility.Vector3dVector(color_arr)

        o3d.visualization.draw_geometries([o3d_point_cloud, mesh_frame, arrow_mesh, voxel_viz_pcd])

    # Key 1: scene (store meta info like ground_R)
    # Key 2-n: object (object idx, operations)
    editing_modifier_dict = {
        "scene": {
            "ground_R": ground_R,
            "ground_T": ground_T,
        },
        "objects": [
            # {
            #     "name": ','.join(fg_obj_list),
            #     "affected_gaussian_idx": final_obj_flag,
            #     "actions": [
            #         {
            #             "action": "rotate",
            #             "rotation": rot_mat,
            #         },
            #         {
            #             "action": "translate",
            #             "translation": translate_vec
            #         }
            #     ]
            # }
            {
                "name": ','.join(fg_obj_list),
                "affected_gaussian_idx": final_obj_flag,
                "actions": [
                    {
                        "action": "physics",
                        "particle_type": "elastic",
                        "infilling_surface_pts": pts_on_disk_n3,
                        "static_idx": rigid_obj_similarity if rigid_object_name else None
                    }
                ]
            },
            # {
            #     "name": BG_OBJ_NAME,
            #     "affected_gaussian_idx": bg_obj_idx,
            #     "actions": [
            #         {
            #             "action": "remove",
            #         }
            #     ]
            # }
        ]
    }

    editing_modifier_save_path = os.path.join(ply_editing_dir, "editing_modifier.pkl")
    with open(editing_modifier_save_path, "wb") as f:
        pickle.dump(editing_modifier_dict, f)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--fg_obj_list", default="vase,flowers,plants", type=str)
    parser.add_argument("--bg_obj_list", default="tabletop,wooden table", type=str)
    parser.add_argument("--ground_plane_name", default="tabletop", type=str)
    parser.add_argument("--rigid_object_name", default="", type=str)
    parser.add_argument("--threshold", default=0.6, type=float)
    parser.add_argument("--object_select_eps", default=0.1, type=float)
    parser.add_argument("--inward_bbox_offset", default=99, type=float, help="Offset for selecting particles inward. Recommended value: 99 (no selection) or 0.1 (select some particles)")
    parser.add_argument("--final_noise_filtering", action="store_true")
    parser.add_argument("--interactive_viz", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    fg_obj_list = args.fg_obj_list.split(",")
    bg_obj_list = args.bg_obj_list.split(",")

    select_gs_for_phys(model.extract(args), args.iteration, fg_obj_list, bg_obj_list, args.ground_plane_name,
                  args.threshold, args.object_select_eps, args.inward_bbox_offset, args.final_noise_filtering,
                  args.interactive_viz, args.rigid_object_name)
