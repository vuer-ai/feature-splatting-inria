import objaverse
import objaverse.xl as oxl
import os
import tempfile
import subprocess

import json
import numpy as np
import shutil
from tqdm import tqdm, trange

RENDER_TIMEOUT = 300

def download_objaverse(uids_list):
    annotations = objaverse.load_annotations(uids_list)

    objects = objaverse.load_objects(uids=uids_list)  # key: uid; value: object path

    return annotations, objects

def blender_render(single_obj_path, render_dir):
    cd_command = 'cd submodules/objaverse_renderer'
    # blender binary
    # 3D model path
    # output directory
    blender_command = '{} --background --python blender_script.py -- --object_path {} --num_renders 200 --output_dir {} --engine BLENDER_EEVEE'

    command = cd_command + ' && ' + blender_command.format(
        'blender-3.2.2-linux-x64/blender',
        single_obj_path,
        render_dir
    )

    subprocess.run(
        ["bash", "-c", command],
        timeout=RENDER_TIMEOUT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def convert_to_nerfstudio_format(blender_rendered_dir, target_dir):
    metadata_path = os.path.join(blender_rendered_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    cam_fov_x = metadata['cam_fov_x']
    cam_fov_y = metadata['cam_fov_y']

    image_paths = [os.path.join(blender_rendered_dir, fn) for fn in os.listdir(blender_rendered_dir) if fn.endswith('.png')]
    image_paths.sort()

    transform_paths = [fn.replace('.png', '.npy') for fn in image_paths]

    os.makedirs(target_dir, exist_ok=True)

    # Copy images
    img_dir = os.path.join(target_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    target_img_rel_path_list = []

    for src_img_path in image_paths:
        dst_img_path = os.path.join(img_dir, os.path.basename(src_img_path))
        target_img_rel_path = os.path.join('images', os.path.basename(src_img_path))
        target_img_rel_path_list.append(target_img_rel_path)
        shutil.copyfile(src_img_path, dst_img_path)

    transforms_json_dict = {}

    transforms_json_dict['camera_angle_x'] = cam_fov_x
    transforms_json_dict['camera_angle_y'] = cam_fov_y
    transforms_json_dict['frames'] = []

    for frame_idx, frame_path in enumerate(image_paths):
        opencv2wld = np.load(transform_paths[frame_idx])  # (3, 4)
        opencv2wld = np.vstack([opencv2wld, [0, 0, 0, 1]])  # (4, 4)
        opencv2wld = np.linalg.inv(opencv2wld)  # (4, 4)

        opengl_ext_mat = opencv2wld

        # Remove .png extension
        rel_path = target_img_rel_path_list[frame_idx]
        assert rel_path.endswith('.png')
        rel_path = rel_path[:-4]

        per_frame_dict = {
            'file_path': rel_path,
            'transform_matrix': opengl_ext_mat.tolist()
        }

        transforms_json_dict['frames'].append(per_frame_dict)

    transforms_train_path = os.path.join(target_dir, 'transforms_train.json')
    transforms_test_path = os.path.join(target_dir, 'transforms_test.json')

    with open(transforms_train_path, 'w') as f:
        json.dump(transforms_json_dict, f, indent=4)

    # TODO(roger): support specifying test frames
    with open(transforms_test_path, 'w') as f:
        json.dump(transforms_json_dict, f, indent=4)

def main(uid_list, target_data_dir):
    os.makedirs(target_data_dir, exist_ok=True)
    # TODO(roger): maybe save some meta information as well?
    print('Downloading objects...')
    annotations, object_path_dict = download_objaverse(uid_list)
    for obj_uid in tqdm(object_path_dict):
        obj_name = annotations[obj_uid]['name']
        obj_path = object_path_dict[obj_uid]
        data_dir = os.path.join(target_data_dir, obj_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            print('Rendering object {}...'.format(obj_name))
            blender_render(obj_path, tmpdir)
            print('Converting to nerf format...')
            convert_to_nerfstudio_format(tmpdir, data_dir)

if __name__ == '__main__':
    uid_list = ['570b82c4391c49ddb1e471e6e55de9f4', 'fbd7fa6a858a4e62a69885e6c3d4a43a']
    target_data_dir = 'feat_data/objaverse'
    main(uid_list, target_data_dir)
