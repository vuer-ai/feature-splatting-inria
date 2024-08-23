/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_COLOR_CHANNELS 3    // Default 3, RGB
#define NUM_FEAT_CHANNELS  16   // Default 768, CLIP ViT; has to be divisible by 2
#define BLOCK_X 16
#define BLOCK_Y 16

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)

#define NUM_GAUSSIAN_LIMIT 100  // Trace the first 10 gaussians

#define FEATURE_SHARED_SIZE 16  // has to be divisible by 2

#endif
