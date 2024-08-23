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

import numpy as np
import torch
import pickle
import torch.nn.functional as F
from scene import Scene
from PIL import Image
import matplotlib as mpl
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.camera_utils import get_current_view
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
from einops import rearrange, einsum
from scene import skip_feat_decoder
import cv2
import featsplat_editor

@torch.no_grad()
def render_interpolating_trajectory(args, model_path, iteration, views, gaussians, pipeline, background, camera_slerp_list=None,
                                    with_feat=False, clip_feat=False, dino_feat=False, with_editing=True, text_query="",
                                    neg_text_query="", step_size=100, part_feat=False):
    if clip_feat or dino_feat:
        assert with_feat
        my_feat_decoder = skip_feat_decoder(gaussians._distill_features.shape[1], part_level=part_feat).cuda()
        decoder_weight_path = os.path.join(model_path, "feat_decoder.pth")
        assert os.path.exists(decoder_weight_path)
        decoder_weight_dict = torch.load(decoder_weight_path)
        my_feat_decoder.load_state_dict(decoder_weight_dict, strict=True)
        my_feat_decoder.eval()
        if text_query:
            clip_segmeter = featsplat_editor.clip_segmenter(gaussians, my_feat_decoder)
            word_list = clip_segmeter.canonical_words.copy() + [neg_text_query] + [text_query]
            text_features_mc = clip_segmeter.get_text_embeddings(word_list).cuda()
        if args.conditional_text_query:
            assert text_query
            conditional_text_features_mc = clip_segmeter.get_text_embeddings(["object", args.conditional_text_query]).cuda()
        
    render_path = os.path.join(model_path,
                               'interpolating_camera',
                               "ours_{}".format(iteration),
                               "renders")

    makedirs(render_path, exist_ok=True)

    total_step_counter = 0

    if with_editing:
        modifier_dict_pt = os.path.join(model_path,
                                       "point_cloud",
                                       "iteration_" + str(iteration),
                                       "editing_modifier.pkl")
        assert os.path.exists(modifier_dict_pt)
        with open(modifier_dict_pt, "rb") as f:
            editing_modifier_dict = pickle.load(f)
        my_gaussian_editor = featsplat_editor.gaussian_editor(editing_modifier_dict)

    assert len(camera_slerp_list) >= 2
    for idx in range(len(camera_slerp_list) - 1):
        start_idx = camera_slerp_list[idx]
        end_idx = camera_slerp_list[idx + 1]
        for i in trange(step_size):
            if with_editing:
                modified_gaussians = my_gaussian_editor.modify_gaussian(gaussians, total_step_counter)
            else:
                modified_gaussians = gaussians
            t = i / step_size
            view = get_current_view(views, start_idx, end_idx, t)
            render_pkg = render(view, modified_gaussians, pipeline, background, render_features=with_feat)
            rendered_rgb = render_pkg["render"]
            save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}'.format(idx, idx + 1, i) + ".png")
            torchvision.utils.save_image(rendered_rgb, save_path)

            if args.render_depth:
                depth_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_depth'.format(idx, idx + 1, i) + ".png")
                depth_np = render_pkg['render_depth'].cpu().numpy().copy()
                depth_viz = depth_np / depth_np.max()
                Image.fromarray((depth_viz * 255).astype(np.uint8), mode='L').save(depth_save_path)

            if with_feat:
                rendered_feat = render_pkg["render_feat"]
                save_principal_feat = F.interpolate(rendered_feat.unsqueeze(0),
                                                    size=(rendered_feat.shape[1], rendered_feat.shape[2]),
                                                    mode='bilinear',
                                                    align_corners=False).squeeze(0)
                principal_feat_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_principal_feat'.format(idx, idx + 1, i) + ".npy")
                np.save(principal_feat_save_path, save_principal_feat.cpu().numpy())
                if clip_feat or dino_feat:
                    # Downsample to save GPU memory
                    # target_H = rendered_feat.shape[1]
                    # target_W = rendered_feat.shape[2]
                    # rendered_feat = F.interpolate(rendered_feat.unsqueeze(0),
                    #                             size=(target_H, target_W),
                    #                             mode='bilinear',
                    #                             align_corners=False)
                    rendered_feat = rendered_feat.unsqueeze(0)
                    rendered_dino_feat, rendered_clip_feat = my_feat_decoder(rendered_feat)
                    rendered_dino_feat = rendered_dino_feat.squeeze(0)
                    rendered_clip_feat = rendered_clip_feat.squeeze(0)
                    # rendered_part_feat = rendered_part_feat.squeeze(0)
                    dino_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_dino_feat'.format(idx, idx + 1, i) + ".npy")
                    clip_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_clip_feat'.format(idx, idx + 1, i) + ".npy")
                    part_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_part_feat'.format(idx, idx + 1, i) + ".npy")
                    # np.save(dino_save_path, rendered_dino_feat.cpu().numpy())
                    # np.save(clip_save_path, rendered_clip_feat.cpu().numpy())

                    if text_query:
                        rendered_clip_feat = rendered_clip_feat / (rendered_clip_feat.norm(dim=0, keepdim=True) + 1e-6)
                        text_probmap = einsum(rendered_clip_feat.float(), text_features_mc.float(), 'c h w, m c -> m h w')
                        text_probmap = text_probmap / 0.01 # Temperature
                        # plt.imshow(normalized_sim > 0.25); plt.show()

                        normalized_sim = text_probmap.softmax(dim=0)[len(word_list) - 1, :, :].float().cpu().numpy()
                        normalized_sim = normalized_sim - normalized_sim.min()
                        normalized_sim = normalized_sim / normalized_sim.max()
                        obj_level_mask = (normalized_sim > 0.6)
                        # normalized_sim = (normalized_sim > 0.6).astype(np.float32)
                        color_arr = mpl.cm.jet(normalized_sim)
                        # color_arr = normalized_sim
                        color_arr = (color_arr * 255).astype(np.uint8)

                        heatmap_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_heatmap'.format(idx, idx + 1, i) + ".png")
                        Image.fromarray(color_arr).save(heatmap_save_path)

                        if args.conditional_text_query:
                            text_probmap = einsum(rendered_clip_feat.float(), conditional_text_features_mc.float(), 'c h w, m c -> m h w')
                            text_probmap = text_probmap / 0.01 # Temperature

                            normalized_sim = text_probmap.softmax(dim=0)[text_probmap.shape[0] - 1, :, :].float().cpu().numpy()
                            normalized_sim[~obj_level_mask] = 0
                            # normalized_sim = normalized_sim - normalized_sim.min()
                            # normalized_sim = normalized_sim / normalized_sim.max()
                            binary_mask = (normalized_sim > 0.6)
                            heatmap_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_obj_con'.format(idx, idx + 1, i) + ".png")
                            color_arr = (binary_mask * 255).astype(np.uint8)
                            Image.fromarray(color_arr).save(heatmap_save_path)

                        if part_feat:
                            rendered_part_feat = rendered_part_feat / (rendered_part_feat.norm(dim=0, keepdim=True) + 1e-6)
                            text_probmap = einsum(rendered_part_feat.float(), text_features_mc.float(), 'c h w, m c -> m h w')
                            text_probmap = text_probmap / 0.1 # Temperature

                            normalized_sim = text_probmap.softmax(dim=0)[len(word_list) - 1, :, :].float().cpu().numpy()
                            color_arr = mpl.cm.jet(normalized_sim)
                            color_arr = (color_arr * 255).astype(np.uint8)

                            part_obj_mask = (normalized_sim > 0.25)
                            part_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_part'.format(idx, idx + 1, i) + ".png")
                            Image.fromarray((part_obj_mask * 255).astype(np.uint8)).save(part_save_path)

                            heatmap_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_part_heatmap'.format(idx, idx + 1, i) + ".png")
                            Image.fromarray(color_arr).save(heatmap_save_path)

                            if args.conditional_text_query:
                                text_probmap = einsum(rendered_part_feat.float(), conditional_text_features_mc.float(), 'c h w, m c -> m h w')
                                text_probmap = text_probmap / 0.01 # Temperature

                                normalized_sim = text_probmap.softmax(dim=0)[text_probmap.shape[0] - 1, :, :].float().cpu().numpy()
                                normalized_sim[~obj_level_mask] = 0
                                binary_mask = mpl.cm.jet(normalized_sim)
                                heatmap_save_path = os.path.join(render_path, '{0}_{1}_step{2:05d}_part_con'.format(idx, idx + 1, i) + ".png")
                                color_arr = (binary_mask * 255).astype(np.uint8)
                                Image.fromarray(color_arr).save(heatmap_save_path)
            
            total_step_counter += 1

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, camera_slerp_list : list,
                with_feat : bool, clip_feat : bool, dino_feat : bool, with_editing : bool, text_query : str,
                neg_text_query : str, step_size : int, args):
    gaussians = GaussianModel(dataset.sh_degree, dataset.distill_feature_dim)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_interpolating_trajectory(args, dataset.model_path, scene.loaded_iter,
                                    scene.getTrainCameras(), gaussians, pipeline,
                                    background, camera_slerp_list, with_feat, clip_feat, dino_feat,
                                    with_editing, text_query, neg_text_query, step_size)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--camera_slerp_list", help="List of camera slerp indices", nargs="+", type=str, default=[0, 0])
    parser.add_argument("--with_feat", action="store_true", default=False)
    parser.add_argument("--clip_feat", action="store_true", default=False)
    parser.add_argument("--dino_feat", action="store_true", default=False)
    parser.add_argument("--with_editing", action="store_true", default=False)
    parser.add_argument("--text_query", help="Text query for CLIP", type=str, default="")
    parser.add_argument("--neg_text_query", help="Negative text query for CLIP", type=str, default="")
    parser.add_argument("--conditional_text_query", help="Conditional text query for CLIP", type=str, default="")
    parser.add_argument("--step_size", help="Step size for camera slerp", type=int, default=100)
    parser.add_argument("--render_depth", action="store_true", default=False)
    parser.add_argument("--render_normal", action="store_true", default=False)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if args.text_query:
        assert args.clip_feat

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.camera_slerp_list,
                args.with_feat, args.clip_feat, args.dino_feat, args.with_editing, args.text_query,
                args.neg_text_query, args.step_size, args)
