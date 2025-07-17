import os
from utils.general_utils import pytorch_gc
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm, trange
import cv2
from typing import Any, Dict, Generator,List
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import maskclip_onnx

# Rate limit workaround (https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511)
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

def resize_image(img, longest_edge):
    # resize to have the longest edge equal to longest_edge
    width, height = img.size
    if width > height:
        ratio = longest_edge / width
    else:
        ratio = longest_edge / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return img.resize((new_width, new_height), Image.BILINEAR)

def interpolate_to_patch_size(img_bchw, patch_size):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W

def is_valid_image(filename):
    ext_test_flag = any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])
    is_file_flag = os.path.isfile(filename)
    return ext_test_flag and is_file_flag
    
def show_anns(anns):
    if len(anns) == 0:
        return
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    return img

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

class MaskCLIPFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(
            "ViT-L/14@336px",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

def main(args):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    yolo_iou = 0.9
    yolo_conf = 0.4

    # For part-level CLIP
    transform = T.Compose([
        T.Resize((args.part_resolution, args.part_resolution)),
        T.ToTensor(),
        norm
    ])

    # For object-level CLIP
    raw_transform = T.Compose([
        T.ToTensor(),
        norm
    ])

    dino_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    clip_model = MaskCLIPFeaturizer().cuda().eval()

    mobilesamv2, ObjAwareModel, predictor = torch.hub.load("RogerQi/MobileSAMV2", args.mobilesamv2_encoder_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()

    base_dir = args.source_path
    image_dir = os.path.join(base_dir, 'images')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(base_dir, 'color')
    assert os.path.isdir(image_dir), f"Image directory {image_dir} does not exist."
    obj_clip_feat_dir = os.path.join(base_dir, 'sam_clip_features')
    os.makedirs(obj_clip_feat_dir, exist_ok=True)
    part_clip_feat_dir = os.path.join(base_dir, 'part_level_features')
    os.makedirs(part_clip_feat_dir, exist_ok=True)
    dinov2_feat_dir = os.path.join(base_dir, 'dinov2_vits14')
    os.makedirs(dinov2_feat_dir, exist_ok=True)

    image_paths = [os.path.join(image_dir, fn) for fn in os.listdir(image_dir)]
    image_paths = [fn for fn in image_paths if is_valid_image(fn)]
    image_paths.sort()

    assert len(image_paths) > 0, f"No valid images found in {image_dir}."
    print(f"Found {len(image_paths)} images.")

    obj_feat_path_list = []
    part_feat_path_list = []
    dinov2_feat_path_list = []
    for image_path in image_paths:
        feat_fn = os.path.splitext(os.path.basename(image_path))[0] + '.npy'
        obj_feat_path = os.path.join(obj_clip_feat_dir, feat_fn)
        part_feat_path = os.path.join(part_clip_feat_dir, feat_fn)
        dinov2_feat_path = os.path.join(dinov2_feat_dir, feat_fn)
        obj_feat_path_list.append(obj_feat_path)
        part_feat_path_list.append(part_feat_path)
        dinov2_feat_path_list.append(dinov2_feat_path)
    
    print("Loading DINOv2 model...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2 = dinov2.to(device)

    for i in trange(len(image_paths)):
        image = Image.open(image_paths[i])
        image = resize_image(image, args.dino_resolution)
        image = dino_transform(image)[:3].unsqueeze(0)
        image, target_H, target_W = interpolate_to_patch_size(image, dinov2.patch_size)
        image = image.cuda()
        with torch.no_grad():
            features = dinov2.forward_features(image)["x_norm_patchtokens"][0]

        features = features.cpu().numpy()

        features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
        features_chw = features_hwc.transpose((2, 0, 1))

        np.save(dinov2_feat_path_list[i], features_chw)
    
    del dinov2
    pytorch_gc()

    # ======================
    tmp_idx = 0
    for i in trange(len(image_paths)):
        image_file_path = image_paths[i]

        image = cv2.imread(image_file_path)
        # resize to longest edge
        if max(image.shape[:2]) > args.sam_size:
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (int(args.sam_size * image.shape[1] / image.shape[0]), args.sam_size))
            else:
                image = cv2.resize(image, (args.sam_size, int(args.sam_size * image.shape[0] / image.shape[1])))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_img_H, raw_img_W = image.shape[:2]

        # part level
        small_W = args.part_feat_res
        small_H = raw_img_H * small_W // raw_img_W

        # obj level
        object_W = args.obj_feat_res
        object_H = raw_img_H * object_W // raw_img_W

        final_W = args.final_feat_res
        final_H = raw_img_H * final_W // raw_img_W

        # ===== Object-aware Model =====
        obj_results = ObjAwareModel(image, device=device, imgsz=args.sam_size, conf=yolo_conf, iou=yolo_iou, verbose=False)

        predictor.set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda()
        sam_mask=[]
        image_embedding=predictor.features
        image_embedding=torch.repeat_interleave(image_embedding, 320, dim=0)
        prompt_embedding=mobilesamv2.prompt_encoder.get_dense_pe()
        prompt_embedding=torch.repeat_interleave(prompt_embedding, 320, dim=0)
        for (boxes,) in batch_iterator(320, input_boxes):
            with torch.no_grad():
                image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
                prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
                sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,)
                low_res_masks, _ = mobilesamv2.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=prompt_embedding,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    simple_type=True,
                )
                low_res_masks=predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)
                sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold)*1.0
                sam_mask.append(sam_mask_pre.squeeze(1))

        sam_mask=torch.cat(sam_mask)
        # Visualize SAM mask
        annotation = sam_mask
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        show_img = annotation[sorted_indices]
        ann_img = show_anns(show_img)
        save_img_path = obj_feat_path_list[i].replace('.npy', '_mask.png')
        Image.fromarray((ann_img * 255).astype(np.uint8)).save(save_img_path)

        # ===== Object-level CLIP feature =====
        raw_input_image = raw_transform(Image.fromarray(image))
        whole_image_feature = clip_model(raw_input_image[None].cuda())[0]
        clip_feat_shape = whole_image_feature.shape[0]

        # Interpolate CLIP features to image size
        resized_clip_feat_map_bchw = torch.nn.functional.interpolate(whole_image_feature.unsqueeze(0).float(),
                                                                size=(object_H, object_W),
                                                                mode='bilinear',
                                                                align_corners=False)

        mask_tensor_bchw = sam_mask.unsqueeze(1)

        resized_mask_tensor_bchw = torch.nn.functional.interpolate(mask_tensor_bchw.float(),
                                                                size=(object_H, object_W),
                                                                mode='nearest').bool()

        aggregated_feat_map = torch.zeros((clip_feat_shape, object_H, object_W), dtype=float, device=device)
        aggregated_feat_cnt_map = torch.zeros((object_H, object_W), dtype=int, device=device)

        for mask_idx in range(resized_mask_tensor_bchw.shape[0]):
            aggregared_clip_feat = resized_clip_feat_map_bchw[0, :, resized_mask_tensor_bchw[mask_idx, 0]]
            aggregared_clip_feat = aggregared_clip_feat.mean(dim=1)

            aggregated_feat_map[:, resized_mask_tensor_bchw[mask_idx, 0]] += aggregared_clip_feat[:, None]
            aggregated_feat_cnt_map[resized_mask_tensor_bchw[mask_idx, 0]] += 1
            
        aggregated_feat_map = aggregated_feat_map / (aggregated_feat_cnt_map[None, :, :] + 1e-6)
        aggregated_feat_map = F.interpolate(aggregated_feat_map[None], (final_H, final_W), mode='bilinear', align_corners=False)[0]

        np.save(obj_feat_path_list[i], aggregated_feat_map.cpu().detach().numpy())

        # visualize bbox
        viz_img = image.copy()
        for bbox_idx in range(input_boxes1.shape[0]):
            bbox = input_boxes1[bbox_idx]
            bbox_xyxy = bbox.cpu().numpy().astype(int)
            cv2.rectangle(viz_img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0), 2)
            cv2.putText(viz_img, f'{bbox_idx}', (bbox_xyxy[0], bbox_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        save_img_path = obj_feat_path_list[i].replace('.npy', '_bbox.png')
        Image.fromarray(viz_img).save(save_img_path)

        # crop images from bbox
        cropped_image_list = []
        bbox_xyxy_list = []
        for bbox_idx in range(input_boxes1.shape[0]):
            bbox = input_boxes1[bbox_idx]
            bbox_xyxy = bbox.cpu().numpy().astype(int)
            bbox_xyxy_list.append(bbox_xyxy)
            crop_img = image[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
            cropped_image_list.append(crop_img)

        image_tensor_list = []
        for cropped_image in cropped_image_list:
            if not isinstance(cropped_image, Image.Image):
                cropped_image = Image.fromarray(cropped_image)
            w, h = cropped_image.size
            image_tensor = transform(cropped_image).unsqueeze(0).to(device)
            image_tensor_list.append(image_tensor)

        # ===== Part-level CLIP feature =====
        aggregared_features = []
        for batch_idx in range(0, len(image_tensor_list), args.part_batch_size):
            with torch.no_grad():
                batch = image_tensor_list[batch_idx:batch_idx+args.part_batch_size]
                batch = torch.cat(batch, dim=0)
                features = clip_model(batch)
                aggregared_features.append(features)

        aggregared_features = torch.cat(aggregared_features, dim=0)

        aggregated_feat_map = torch.zeros((clip_feat_shape, small_H, small_W), dtype=float, device=device)
        aggregated_feat_cnt_map = torch.zeros((small_H, small_W), dtype=int, device=device)

        assert len(image_tensor_list) == aggregared_features.shape[0]

        for obj_idx in range(len(image_tensor_list)):
            resized_bbox = (bbox_xyxy_list[obj_idx] * (small_W / image.shape[1])).astype(int)
            feat_h = int(resized_bbox[3] - resized_bbox[1])
            feat_w = int(resized_bbox[2] - resized_bbox[0])

            resized_feature = F.interpolate(aggregared_features[obj_idx].unsqueeze(0), (feat_h, feat_w), mode='bilinear', align_corners=False)[0]
            aggregated_feat_map[:, resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]] += resized_feature

            aggregated_feat_cnt_map[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]] += 1

        aggregated_feat_map = aggregated_feat_map / (aggregated_feat_cnt_map[None,:,:] + 1e-6)
        aggregated_feat_map = F.interpolate(aggregated_feat_map[None], (final_H, final_W), mode='bilinear', align_corners=False)[0]
        aggregated_feat_map = aggregated_feat_map.cpu().numpy()

        np.save(part_feat_path_list[i], aggregated_feat_map)

if __name__ == "__main__":
    parser = ArgumentParser("Compute reference features for feature splatting")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--part_batch_size", type=int, default=32, help="Part-level CLIP inference batch size")
    parser.add_argument("--part_resolution", type=int, default=224, help="Part-level CLIP input image resolution")
    parser.add_argument("--sam_size", type=int, default=1024, help="Longest edge for MobileSAMV2 segmentation")
    parser.add_argument("--obj_feat_res", type=int, default=100, help="Intermediate (for MAP) SAM-enhanced Object-level feature resolution")
    parser.add_argument("--part_feat_res", type=int, default=400, help="Intermediate (for MAP) SAM-enhanced Part-level feature resolution")
    parser.add_argument("--final_feat_res", type=int, default=64, help="Final hierarchical CLIP feature resolution")
    parser.add_argument("--dino_resolution", type=int, default=800, help="Longest edge for DINOv2 feature generation")
    parser.add_argument("--mobilesamv2_encoder_name", type=str, default="mobilesamv2_efficientvit_l2", help="MobileSAMV2 encoder name")
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
