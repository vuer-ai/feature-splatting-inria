# Change Log for feature splatting rasterizer

## 0.0.0

This is the original version from [gaussian splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d).

## 0.0.1

Draft implementation of feature splatting, which contains optimized forward/backward passes implementations.

## 0.0.2

Fixed array allocation and block synchronization issues that caused artifacts in rendering. Fully working feature splatting.

## 0.0.3

Added optional feature rendering flag and remove debugging rendering for faster training.

## 0.0.4

Support depth supervision in FP/BP (thanks to pointer provided by Stephan Yang).
