import time
import numpy as np
import torch

from einops import rearrange, einsum
from submodules.patched_clip.patched_clip.patched_clip import CLIP_args, load_clip
from submodules.patched_clip.patched_clip import modified_clip

class clip_segmenter:
    def __init__(self, gaussians, conv_feat_decoder,
                 canonical_words=['object', 'things'],
                 clip_device='cpu') -> None:
        self.gaussians = gaussians
        self.feat_decoder = conv_feat_decoder

        CLIP_args.device = clip_device
        self.device = CLIP_args.device

        self.clip_model, _ = load_clip()

        self.canonical_words = canonical_words

        # Convert 1x1 conv into a shallow MLP
        # TODO(roger): this should be done in the model definition of feat decoder
        #              otherwise, if a model definition changes, this will break
        weight_dict = conv_feat_decoder.state_dict()

        self.fc = torch.nn.utils.weight_norm(torch.nn.Linear(32, 256))
        self.clip_fc = torch.nn.utils.weight_norm(torch.nn.Linear(256, 768))
        
        self.fc.weight_g = torch.nn.Parameter(weight_dict['conv1.weight_g'][:,:,0,0])
        self.fc.weight_v = torch.nn.Parameter(weight_dict['conv1.weight_v'][:,:,0,0])
        self.fc.bias = torch.nn.Parameter(weight_dict['conv1.bias'])

        self.clip_fc.weight_g = torch.nn.Parameter(weight_dict['clip_conv1.weight_g'][:,:,0,0])
        self.clip_fc.weight_v = torch.nn.Parameter(weight_dict['clip_conv1.weight_v'][:,:,0,0])
        self.clip_fc.bias = torch.nn.Parameter(weight_dict['clip_conv1.bias'])

        self.fc.to(self.device)
        self.clip_fc.to(self.device)

        if 'part_conv1.bias' in weight_dict:
            self.part_clip_fc = torch.nn.utils.weight_norm(torch.nn.Linear(256, 768))
            self.part_clip_fc.weight_g = torch.nn.Parameter(weight_dict['part_conv1.weight_g'][:,:,0,0])
            self.part_clip_fc.weight_v = torch.nn.Parameter(weight_dict['part_conv1.weight_v'][:,:,0,0])
            self.part_clip_fc.bias = torch.nn.Parameter(weight_dict['part_conv1.bias'])

            self.part_clip_fc.to(self.device)

            self.part_level_available = True
        else:
            self.part_level_available = False
    
    def get_text_embeddings(self, texts):
        """
        Get CLIP embeddings for a list of texts.
        """
        with torch.no_grad():
            device = CLIP_args.device
            # text
            text_inputs = torch.cat([modified_clip.tokenize(f"{c}") for c in texts]).to(device)

            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=1, keepdim=True)

            return text_features
    
    def decoder_infer(self, x, level="object"):
        assert level in ["object", "part"]
        with torch.no_grad():
            x = self.fc(x)
            x = torch.nn.functional.relu(x)
            if level == "object":
                x = self.clip_fc(x)
            else:
                assert self.part_level_available
                x = self.part_clip_fc(x)
            return x
    
    def compute_similarity_on_downsampled(self, positive_text_query, negative_text_query=[], sample_size=2**15, level="object", threshold=0.5, feature=None):
        """
        Computer normalized similarity of gaussians using one word w.r.t. canonical words.

        This only provide a very rough localization of the object, which can be used to trim down the search space.
        """
        if isinstance(positive_text_query, str):
            positive_text_query = [positive_text_query]
        if isinstance(negative_text_query, str):
            negative_text_query = [negative_text_query]
        
        if feature is None:
            feature = self.gaussians.get_distill_features

        word_list = self.canonical_words.copy() + negative_text_query + positive_text_query
        viz_category_idx = len(word_list) - 1
        text_features_mc = self.get_text_embeddings(word_list)

        # Downsample features
        sample_idx = np.random.choice(feature.shape[0], sample_size, replace=False)
        with torch.no_grad():
            chunk = feature[sample_idx].to(self.device)
            chunk_feature_nc = self.decoder_infer(chunk, level)
            chunk_feature_nc = chunk_feature_nc / (chunk_feature_nc.norm(dim=1, keepdim=True) + 1e-6)
            similarity_nm = einsum(chunk_feature_nc.float(), text_features_mc.float(), 'n c, m c -> n m')

        similarity_nm = similarity_nm / 0.05
        normalized_sim = similarity_nm.softmax(dim=1)[:, -len(positive_text_query):].sum(dim=1)

        assert torch.isnan(normalized_sim).sum() == 0

        normalized_sim = normalized_sim.float().cpu().numpy().flatten()
        normalized_sim -= normalized_sim.min()
        normalized_sim /= normalized_sim.max()
        selected_obj_idx = normalized_sim > threshold

        return selected_obj_idx, sample_idx
    
    def fast_compute_rough_bbox(self, positive_text_query, negative_text_query=[], bbox_edge=2, sample_size=2**15, level="object", threshold=0.5, feature=None, xyz=None):
        """
        Computer normalized similarity of gaussians using one word w.r.t. canonical words.

        This only provide a very rough localization of the object, which can be used to trim down the search space.
        """
        if feature is None:
            feature = self.gaussians.get_distill_features
        if xyz is None:
            xyz = self.gaussians.get_xyz
        selected_obj_idx, sample_idx = self.compute_similarity_on_downsampled(positive_text_query, negative_text_query, sample_size, level, threshold, feature=feature)
        subset_xyz = xyz[sample_idx].cpu().numpy()

        # TODO(roger): the eps may fail if objects are too close
        cluster_instance_idx = self.cluster_instance(subset_xyz, selected_obj_idx)

        selected_xyz = subset_xyz[cluster_instance_idx]
        selected_xyz_min = np.min(selected_xyz, axis=0) - bbox_edge
        selected_xyz_max = np.max(selected_xyz, axis=0) + bbox_edge

        return np.vstack([selected_xyz_min, selected_xyz_max])

    def compute_similarity_one(self, positive_text_query, eps=1e-6, chunk_size=2**14, level="object", feature=None):
        """
        Computer normalized similarity of gaussians using one word w.r.t. canonical words
        """
        if isinstance(positive_text_query, str):
            positive_text_query = [positive_text_query]

        if feature is None:
            feature = self.gaussians.get_distill_features
        word_list = self.canonical_words.copy() + positive_text_query
        viz_category_idx = len(word_list) - 1
        text_features_mc = self.get_text_embeddings(word_list)

        num_trunks = int(np.ceil(feature.shape[0] / chunk_size))
        similarity_nm = []
        for i in range(num_trunks):
            chunk = feature[i*chunk_size:(i+1)*chunk_size].to(self.device)
            chunk_feature_nc = self.decoder_infer(chunk, level)
            chunk_feature_nc = chunk_feature_nc / (chunk_feature_nc.norm(dim=1, keepdim=True) + eps)
            similarity_nm.append(einsum(chunk_feature_nc.float(), text_features_mc.float(), 'n c, m c -> n m'))
            
        similarity_nm = torch.cat(similarity_nm, dim=0)

        similarity_nm = similarity_nm / 0.05
        normalized_sim = similarity_nm.softmax(dim=1)[:, -len(positive_text_query):].sum(dim=1)

        assert torch.isnan(normalized_sim).sum() == 0

        normalized_sim = normalized_sim.float().cpu().numpy().flatten()
        normalized_sim -= normalized_sim.min()
        normalized_sim /= normalized_sim.max()

        return normalized_sim
    
    def knn_dilation(self, all_xyz_n3, obj_idx, k=50, dilation=0.1, dilation_iters=3, positive_ratio=0.8):
        """
        Use KD Tree to add background points whose neighbors are (mostly) foreground points
        """
        from pykdtree.kdtree import KDTree
        obj_idx = obj_idx.copy()
        for _ in range(dilation_iters):
            fg_xyz = all_xyz_n3[obj_idx]
            non_fg_xyz = all_xyz_n3[~obj_idx]
            fg_kdtree = KDTree(fg_xyz)
            dist_nk, indices_nk = fg_kdtree.query(non_fg_xyz, k=k)
            # TODO(roger): this dilation param needs to be tuned due to scale ambiguity
            in_range_count = np.sum(dist_nk < dilation, axis=1)
            non_fg_indices = np.arange(all_xyz_n3.shape[0])[~obj_idx]
            non_fg_indices = non_fg_indices[in_range_count > int(k * positive_ratio)]
            obj_idx[non_fg_indices] = True
        return obj_idx
    
    def knn_infilling(self, all_xyz_n3, obj_idx, k=50, dilation_iters=3, positive_ratio=0.8):
        """
        Use KD Tree to add background points whose neighbors are (mostly) foreground points
        """
        from pykdtree.kdtree import KDTree
        obj_idx = obj_idx.copy()
        for _ in range(dilation_iters):
            # fg_xyz = all_xyz_n3[obj_idx]
            non_fg_xyz = all_xyz_n3[~obj_idx]
            fg_kdtree = KDTree(all_xyz_n3)
            dist_nk, indices_nk = fg_kdtree.query(non_fg_xyz, k=k)
            positive_cnt = np.sum(obj_idx[indices_nk], axis=1)
            non_fg_indices = np.arange(all_xyz_n3.shape[0])[~obj_idx]
            non_fg_indices = non_fg_indices[positive_cnt > int(k * positive_ratio)]
            obj_idx[non_fg_indices] = True
        return obj_idx
    
    def cluster_instance(self, all_xyz_n3, selected_obj_idx=None, min_sample=20, eps=0.1):
        """
        Cluster points into instances using DBSCAN.
        Return the indices of the most populated cluster.
        """
        from sklearn.cluster import DBSCAN
        if selected_obj_idx is None:
            selected_obj_idx = np.ones(all_xyz_n3.shape[0], dtype=bool)
        dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(all_xyz_n3[selected_obj_idx])
        clustered_labels = dbscan.labels_

        # Find the most populated cluster
        label_idx_list, label_count_list = np.unique(clustered_labels, return_counts=True)
        # Filter out -1
        label_count_list = label_count_list[label_idx_list != -1]
        label_idx_list = label_idx_list[label_idx_list != -1]
        max_count_label = label_idx_list[np.argmax(label_count_list)]

        clustered_idx = np.zeros_like(selected_obj_idx, dtype=bool)
        # Double assignment to make sure indices go into the right place
        arr = clustered_idx[selected_obj_idx]
        arr[clustered_labels == max_count_label] = True
        clustered_idx[selected_obj_idx] = arr
        return clustered_idx

    def ground_bbox_filter(self, all_xyz_n3, selected_obj_idx, ground_R, ground_T, boundary,
                           skip_upwards=False, less_upwards=False):
        """
        Select points within a bounding box.
        """
        particles = all_xyz_n3 @ ground_R.T
        particles += ground_T
        xyz_min = np.min(particles[selected_obj_idx], axis=0)
        xyz_max = np.max(particles[selected_obj_idx], axis=0)
        xyz_min += boundary
        xyz_max -= boundary
        if skip_upwards:
            xyz_max[1] += boundary[1] * 2
        if less_upwards:
            xyz_max[1] -= 2.0
        bbox_particles_idx = np.all((particles > xyz_min) & (particles < xyz_max), axis=1)
        assert bbox_particles_idx.shape[0] == selected_obj_idx.shape[0]
        bbox_selected_particles = np.logical_or(bbox_particles_idx, selected_obj_idx)
        return bbox_selected_particles
    
    def remove_ground(self, all_xyz_n3, selected_obj_idx, ground_R, ground_T, ground_level=0):
        """
        Remove points that are (obviously) below the ground plane.

        Ground level is default to 0 because ground transformation solved by RANSAC usually is *slightly*
        below the actual ground contact surface.
        """
        particles = all_xyz_n3 @ ground_R.T
        particles += ground_T
        non_ground_idx = particles[:, 1] > ground_level
        bbox_selected_particles = np.logical_and(non_ground_idx, selected_obj_idx)
        return bbox_selected_particles
