import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from sklearn.decomposition import PCA

def main(base_dir, save_dir, no_fast_pca, mask_zeros):
    # Get a list of one or more feature npy files
    feature_path_list = [fn for fn in os.listdir(base_dir) if fn.endswith('.npy')]
    feature_path_list = [os.path.join(base_dir, fn) for fn in feature_path_list]
    feature_path_list = sorted(feature_path_list)

    os.makedirs(save_dir, exist_ok=True)

    all_feature_list = []

    print("Computing PCA with {} features".format(len(feature_path_list)))
    # TODO(roger): this takes a long time. We can use a faster PCA that uses a random subset of the data
    for idx in trange(0, len(feature_path_list), 10):
        feature_chw = np.load(feature_path_list[idx])
        assert len(feature_chw.shape) == 3
        C, H, W = feature_chw.shape
        feature_nc = feature_chw.reshape((C, -1)).transpose((1, 0))
        all_feature_list.append(feature_nc)

    feature_mc = np.vstack(all_feature_list)
    feature_mc_mask = np.random.choice(feature_mc.shape[0], 50, replace=False)
    feature_mc = feature_mc[feature_mc_mask]

    feature_mc[np.isnan(feature_mc)] = 0

    if mask_zeros:
        # Keep contain only features that have less than 40 zeros
        feature_mc_mask = (feature_mc < 1e-2).sum(axis=1) < (C * 0.8)
        feature_mc = feature_mc[feature_mc_mask]

    pca = PCA(n_components=3)
    X = pca.fit_transform(feature_mc[::10])

    # Use 10th and 90th percentile for min and max so the feature viz is brighter
    quan_min_X = np.quantile(X, 0.05)
    quan_max_X = np.quantile(X, 0.95)

    print("Saving PCA features")
    for idx in trange(len(feature_path_list)):
        feature_chw = np.load(feature_path_list[idx])
        feature_chw[np.isnan(feature_chw)] = 0
        assert len(feature_chw.shape) == 3
        C, H, W = feature_chw.shape
        feature_nc = feature_chw.reshape((C, -1)).transpose((1, 0))
        feature_3c = pca.transform(feature_nc)
        feature_3c = (feature_3c - quan_min_X) / (quan_max_X - quan_min_X) * 255
        feature_3c = np.clip(feature_3c, 0, 255)
        feature_3c = np.uint8(feature_3c)
        feature_rgb = feature_3c.reshape((H, W, 3))
        
        if mask_zeros:
            invalid_feature_mask = (feature_chw < 1e-2).sum(axis=0) > (C * 0.8)
            print("Invalid feature mask: ", invalid_feature_mask.sum())
            feature_rgb[invalid_feature_mask] = 255

        # Make it RGBA
        feature_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        feature_rgba[..., :3] = feature_rgb
        feature_rgba[..., 3] = 255
        if mask_zeros:
            feature_rgba[invalid_feature_mask, 3] = 0
        
        save_fn = os.path.basename(feature_path_list[idx]).replace('.npy', '.png')
        save_path = os.path.join(save_dir, save_fn)
        Image.fromarray(feature_rgba).save(save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--no_fast_pca', action='store_true')
    parser.add_argument('--mask_zeros', action='store_true')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.no_fast_pca, args.mask_zeros)
