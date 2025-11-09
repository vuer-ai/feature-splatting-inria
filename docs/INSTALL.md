## System-level Prereq

If you don't have conda/miniconda, I recommend miniconda, which is a lightweight version of anaconda that has essential features.

You can get the installation [here](https://docs.anaconda.com/free/miniconda/).

Follow the instruction [here](https://stackoverflow.com/questions/76760906/installing-mamba-on-a-machine-with-conda) to set up Mamba, a fast environment solver for conda.

```
## prioritize 'conda-forge' channel
conda config --add channels conda-forge

## update existing packages to use 'conda-forge' channel
conda update -n base --all

## install 'mamba'
conda install -n base mamba
```

Note: technically, the mamba solver should behave the same as the default solver. However, there have been cases where dependencies
can not be properly set up with the default mamba solver. The following instructions have **only** been tested on mamba solver.

### Install Gaussian-related packages

```
conda create -y -n feature_splatting python=3.11 && conda activate feature_splatting
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit
git clone --recursive https://github.com/vuer-ai/feature-splatting-inria
cd feature-splatting-inria
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install -r requirements.txt
```

### FAQ

1. If you encounter the following error:
```
ImportError: cannot import name 'packaging' from 'pkg_resources' 
```

It may be due to the version of setuptools. You can downgrade the version of setuptools by running the following command:
```
pip install setuptools==69.5.1
```

2. Weird error such as `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 263168.04 GiB.`

This has something to do with the compute capability specified in [rasterizer setup config](../submodules/diff-gaussian-rasterization/setup.py).
You may need to set compute capability for your GPU, especially if it is a newer release.