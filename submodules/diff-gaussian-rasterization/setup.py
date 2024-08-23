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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

# Set environment variable for CUDA compute capability
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6 8.0 7.0"

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    version="0.0.4",
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ['-Xptxas="-v"', "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# === Might be useful if we need dynamic linking in the future ===
# setup(
#     name="diff_gaussian_rasterization",
#     packages=['diff_gaussian_rasterization'],
#     ext_modules=[
#         CUDAExtension(
#             name="diff_gaussian_rasterization._C",
#             sources=[
#             "cuda_rasterizer/rasterizer_impl.cu",
#             "cuda_rasterizer/forward.cu",
#             "cuda_rasterizer/backward.cu",
#             "rasterize_points.cu",
#             "ext.cpp"],
#             dlink=True,
#             dlink_libraries=["dlink_lib"],
#             extra_compile_args={"nvcc": ['-lcudadevrt', "-dc", "-code=lto_86", "-arch=compute_86", "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
#         ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )