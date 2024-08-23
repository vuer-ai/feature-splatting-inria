# Processing your own data

Processing data for feature splatting is similar to Gaussian splatting. Currently, we only support colmap format data.

## Colmap

To use colmap, first make sure colmap is available on your machine by running `colmap -h` and checking that
help message is printed correctly. If not, please follow the instructions [here](https://colmap.github.io/install.html).

Note: colmap works with or without CUDA. CUDA is automatically detected in colmap installation process by checking if nvcc is available.
However, sometimes colmap can run into trouble with CUDA installed in conda. In this case, please run `conda deactivate` and return to
the base environment before compiling colmap.

With installed colmap, let's say we create a `sample_dataset` under the `feat_data` folder. The user
can put images into the input folder under the `sample_dataset` (i.e., `feat_data/sample_data/input`). Then we can run,

```bash
python convert.py -s feat_data/sample_data
```

to estimate information such as camera intrinsics/extrinsics and SfM sparse point clouds that help with Gaussian training.

Then we can compute features via

```bash
python compute_obj_part_feature.py -s feat_data/sample_data
```
