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
conda create -y -n feature_splatting python=3.8 && conda activate feature_splatting
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
git clone --recursive https://github.com/RogerQi/gaussian_feature
cd gaussian_feature
cd submodules/diff-gaussian-rasterization
pip install .
cd ../..
cd submodules/simple-knn
pip install .
cd ../..
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
