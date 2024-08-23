#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from scene.cameras import Camera
from copy import deepcopy
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import struct

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    largest_width_allowed = 800

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > largest_width_allowed:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>800 pixels width), rescaling to 800.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / largest_width_allowed
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]  # (3, H, W)
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # TODO(roger): make feature loading optional and support specifying feature
    img_path = cam_info.image_path
    img_base_dir = os.path.dirname(img_path)
    feat_fn = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
    feat_path = os.path.join(img_base_dir, '..', 'sam_clip_features', feat_fn)
    dino_feat_path = os.path.join(img_base_dir, '..', 'dinov2_vits14', feat_fn)
    part_feat_path = os.path.join(img_base_dir, '..', 'part_level_features', feat_fn)

    if not os.path.exists(feat_path):
        print("Feature path {} does not exist".format(feat_path))
        feat_np_chw = np.zeros(1)
    else:
        feat_np_chw = np.load(feat_path)
    feat_chw = torch.from_numpy(feat_np_chw).float()

    if not os.path.exists(dino_feat_path):
        print("DINO feature path {} does not exist".format(dino_feat_path))
        dino_feat_np_chw = np.zeros(1)
    else:
        dino_feat_np_chw = np.load(dino_feat_path)
    dino_feat_chw = torch.from_numpy(dino_feat_np_chw).float()

    if not os.path.exists(part_feat_path):
        print("Part feature path {} does not exist".format(part_feat_path))
        part_feat_np_chw = np.zeros(1)
    else:
        part_feat_np_chw = np.load(part_feat_path)
    part_feat_chw = torch.from_numpy(part_feat_np_chw).float()

    depth_fn = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    depth_path = os.path.join(img_base_dir, '..', 'depth', depth_fn)
    depth_scaling_factor = None
    if not os.path.exists(depth_path):
        # change suffix to .npy
        depth_fn = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        depth_path = os.path.join(img_base_dir, '..', 'depth', depth_fn)
        if not os.path.exists(depth_path):
            depth = None
        else:
            depth_np = np.load(depth_path)
            depth_pil = Image.fromarray(depth_np)
            # Resize to match the image size
            depth_pil = depth_pil.resize((gt_image.shape[2], gt_image.shape[1]), Image.NEAREST)
            depth_np = np.array(depth_pil, dtype=np.float32)
            depth = torch.from_numpy(depth_np).float()
    else:
        depth_np = Image.open(depth_path)
        # Resize to match the image size
        depth_np = depth_np.resize((gt_image.shape[2], gt_image.shape[1]), Image.NEAREST)
        depth_np = np.array(depth_np)
        depth = torch.from_numpy(depth_np).float()
        depth_scaling_factor = 1000.0
    
    mask_fn = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    mask_path = os.path.join(img_base_dir, '..', 'render_masks', mask_fn)
    if not os.path.exists(mask_path):
        mask = None
    else:
        mask_np = Image.open(mask_path)
        mask_np = mask_np.resize((gt_image.shape[2], gt_image.shape[1]), Image.NEAREST)
        mask_np = np.array(mask_np)
        mask_np = (mask_np > 0)
        mask = torch.from_numpy(mask_np).bool()

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  feat_path=feat_path, feat_chw=feat_chw, dino_feat_path=dino_feat_path, dino_feat_chw=dino_feat_chw,
                  depth=depth, mask=mask, depth_scaling_factor=depth_scaling_factor, part_feat_chw=part_feat_chw)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def manual_slerp(quat1, quat2, t):
    """
    Manual implementation of Spherical Linear Interpolation (SLERP) for quaternions.
    """
    dot_product = np.dot(quat1, quat2)

    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot_product < 0.0:
        quat1 = -quat1
        dot_product = -dot_product
    
    # Clamp value to stay within domain of acos()
    dot_product = np.clip(dot_product, -1.0, 1.0)

    theta_0 = np.arccos(dot_product)  # angle between input vectors
    theta = theta_0 * t  # angle between v0 and result

    quat2 = quat2 - quat1 * dot_product
    quat2 = quat2 / np.linalg.norm(quat2)

    return quat1 * np.cos(theta) + quat2 * np.sin(theta)

def interpolate_se3(T1, T2, t):
    """
    Interpolates between two SE(3) poses.
    
    :param T1: First SE(3) matrix.
    :param T2: Second SE(3) matrix.
    :param t: Interpolation factor (0 <= t <= 1).
    :return: Interpolated SE(3) matrix.
    """
    if np.isclose(T1 - T2, 0).all():
        return T1
    # Decompose matrices into rotation (as quaternion) and translation
    rot1, trans1 = T1[:3, :3], T1[:3, 3]
    rot2, trans2 = T2[:3, :3], T2[:3, 3]
    quat1, quat2 = R.from_matrix(rot1).as_quat(), R.from_matrix(rot2).as_quat()

    # Spherical linear interpolation (SLERP) for rotation
    # Manual SLERP for rotation
    interp_quat = manual_slerp(quat1, quat2, t)
    interp_rot = R.from_quat(interp_quat).as_matrix()

    # Linear interpolation for translation
    interp_trans = interp1d([0, 1], np.vstack([trans1, trans2]), axis=0)(t)

    # Recompose SE(3) matrix
    T_interp = np.eye(4)
    T_interp[:3, :3] = interp_rot
    T_interp[:3, 3] = interp_trans

    return T_interp

def interpolate_camera_se3(view1, view2, t):
    """
    Interpolates between two Camera poses.
    
    :param view1: First Camera.
    :param view2: Second Camera.
    :param t: Interpolation factor (0 <= t <= 1).
    :return: Interpolated Camera.
    """
    view = deepcopy(view1)
    view.world_view_transform = torch.tensor(
        interpolate_se3(
            view1.world_view_transform.cpu().numpy().T,
            view2.world_view_transform.cpu().numpy().T,
            t).T
    ).cuda().float()
    view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]
    return view

def load_cam_from_SIBR_binary(filename):
    """
    Read .bin camera view file saved by SIBR viewer, and convert it into the coordinate system of 3DGS.

    This function is the python implementation of the util function invoked by "save camera(bin)" in SIBR viewer IMGUI at
    https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_openxr/src/core/assets/InputCamera.cpp?ref_type=heads#L1107
    """
    # Hardwired version number from
    # https://gitlab.inria.fr/sibr/sibr_core/-/blob/a89f31c99ba332f047b3b53bfbe60b0b496b874e/src/core/assets/InputCamera.cpp#L22
    SIBR_INPUTCAMERA_BINARYFILE_VERSION = 10

    with open(filename, 'rb') as f:
        data = f.read()

    offset = 0

    # Read version
    version = struct.unpack_from('B', data, offset)[0]
    offset += 1

    assert version == SIBR_INPUTCAMERA_BINARYFILE_VERSION

    # Read remaining data
    (focal, k1, k2, w, h) = struct.unpack_from('>fffHH', data, offset)
    offset += struct.calcsize('>fffHH')
    pos = struct.unpack_from('>fff', data, offset)
    offset += struct.calcsize('>fff')
    rot = struct.unpack_from('>ffff', data, offset)
    offset += struct.calcsize('>ffff')
    (fov, aspect, znear, zfar) = struct.unpack_from('ffff', data, offset)

    # Camera extrinsics are given in SIBR's coordinate system
    position = np.array(pos)
    rotation = np.array(rot)

    # Variables that we don't use now
    _focal = focal
    _k1 = k1
    _k2 = k2
    _w = w
    _h = h
    fovy = fov
    aspect = aspect
    znear = znear
    zfar = zfar

    position = np.array(position)
    rotation = np.array(rotation)

    # Conversion matrix from SIBR is given at
    # https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_openxr/src/core/assets/InputCamera.cpp?ref_type=heads#L1377
    conversion_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    r = R.from_quat(rotation[[1, 2, 3, 0]])
    sibr_rot = r.as_matrix()

    colmap_rot = sibr_rot @ conversion_matrix.T

    colmap_r = R.from_matrix(colmap_rot.T)
    colmap_quat = colmap_r.as_quat()
    colmap_rot_mat = colmap_r.as_matrix()

    sibr_position = position

    colmap_position = -sibr_position
    colmap_position = sibr_rot.T @ colmap_position
    colmap_position = conversion_matrix.T @ colmap_position

    # gs_position and gs_rot_mat are consistent with camera class from rendering and training
    gs_position = colmap_position
    gs_rot_mat = colmap_rot_mat.T

    return (gs_rot_mat, gs_position)

def get_single_view(views, idx):
    if idx.isnumeric():
        idx = int(idx)
        return views[idx]
    elif idx.endswith('.bin'):
        R, T = load_cam_from_SIBR_binary(idx)
        view = deepcopy(views[0])
        view.set_new_transform(R, T)
        return view
    else:
        raise NotImplementedError("Got unsupported view index {}".format(idx))

def get_current_view(views, start_idx, end_idx, t):
    start_view = get_single_view(views, start_idx)
    end_view = get_single_view(views, end_idx)
    current_view = interpolate_camera_se3(start_view, end_view, t)
    return current_view
