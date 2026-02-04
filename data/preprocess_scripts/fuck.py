import os
import numpy as np
import torch
import open3d as o3d
from smplx import SMPL
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from utils.colmap import read_cameras_text, read_frames_text, read_images_text, read_points3D_text, rescale_colmap_dataset, get_scaled_smpl_dict
from utils.graphics import get_intrinsic_matrix, get_extrinsic_matrix, transform_to_world_space, auto_align_to_floor, convert_smpl_dict_to_world, get_smpl_vertice
from utils.visualize import visualize_projection

smpl_model = SMPL(model_path="../SMPL_NEUTRAL.pkl", gender='neutral')



DATASETS=[
    "P0_08_outdoor_remove_jacket",
    "P2_23_outdoor_hug_tree",
    "P3_32_outdoor_soccer_warmup_a",
    "P5_42_indoor_dancing",
    "P6_50_outdoor_workout",
    "P7_61_outdoor_sit_lie_walk",
    "P8_67_outdoor_workout_stretch",
]

for dataset_name in DATASETS:
    print(dataset_name)
    cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
    images = read_images_text(f'../{dataset_name}/sparse/0/images.txt')
    points3D = read_points3D_text(f'../{dataset_name}/sparse/0/points3D.txt')

    images_dir = f'../{dataset_name}/images'
    image_files = sorted(os.listdir(images_dir))
    image_name_0 = image_files[0]
    index = int(image_name_0[:-4])
    print(image_name_0, index)
    cam_extrinsic = get_extrinsic_matrix(images[index])
    smpl_data = dict(np.load(f'../{dataset_name}/sam3d/smpl/smpl_{index:05d}.npz', allow_pickle=True))
    smpl_vertice_cam = smpl_model(betas=torch.from_numpy(smpl_data['betas'])[None], body_pose=torch.from_numpy(smpl_data['body_pose'])[None],global_orient=torch.from_numpy(smpl_data['global_orient'])[None], transl=torch.from_numpy(smpl_data['transl'])[None]).vertices.detach().cpu().numpy().squeeze()
    smpl_vertice_world = transform_to_world_space(smpl_vertice_cam, cam_extrinsic)
    smpl_vertice_cam = np.hstack([smpl_vertice_cam, np.full_like(smpl_vertice_cam, 128)])
    smpl_vertice_world = np.hstack([smpl_vertice_world, np.full_like(smpl_vertice_world, 128)])

    cam_intrinsic = get_intrinsic_matrix(cameras[1])
    image_path = f'../{dataset_name}/images/{image_name_0}'
    H, W = cameras[1]['height'], cameras[1]['width']
    visualize_projection(smpl_vertice_world, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")

    s = auto_align_to_floor(smpl_vertice_cam, points3D, cam_extrinsic)
    rescale_colmap_dataset(f'../{dataset_name}/sparse/0/images.txt', f'../{dataset_name}/sparse/0/points3D.txt', s)


    cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
    images = read_images_text(f'../{dataset_name}/sparse/0/images_rescaled.txt')
    points3D = read_points3D_text(f'../{dataset_name}/sparse/0/points3D_rescaled.txt')
    cam_intrinsic = get_intrinsic_matrix(cameras[1])

    # image_00000
    files_num = len([f for f in os.listdir(f"../{dataset_name}/images") if f.endswith(('.jpg', '.png'))])

    for i in range(files_num):

        images_dir = f'../{dataset_name}/images'
        image_files = sorted(os.listdir(images_dir))
        image_name = image_files[i]
        print(image_name)
        index = int(image_name[:-4])
        print(index)
        image_path = f'../{dataset_name}/images/{image_name}'
        H, W = cameras[1]['height'], cameras[1]['width']
        cam_extrinsic = get_extrinsic_matrix(images[index])
        smpl_data_cam = dict(np.load(f'../{dataset_name}/sam3d/smpl/smpl_{index:05d}.npz', allow_pickle=True))
        smpl_vertice_cam = get_smpl_vertice(smpl_data_cam)
        # smpl_world_1 = transform_to_world_space(smpl_vertice_cam, cam_extrinsic)
        # visualize_projection(smpl_world_1, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")


        def get_template_pelvis(smpl_model, betas):
            if isinstance(betas, np.ndarray):
                betas_torch = torch.from_numpy(betas).float()
            else:
                betas_torch = betas.float()
            if betas_torch.ndim == 1:
                betas_torch = betas_torch.unsqueeze(0)

            device = next(smpl_model.parameters()).device
            betas_torch = betas_torch.to(device)
            with torch.no_grad():
                output = smpl_model(
                    betas=betas_torch,
                    body_pose=torch.zeros((1, 69), device=device),
                    global_orient=torch.zeros((1, 3), device=device),
                    transl=torch.zeros((1, 3), device=device)
                )
            j_root = output.joints[0, 0].detach().cpu().numpy()
            return j_root
        j_root = get_template_pelvis(smpl_model, smpl_data_cam['betas'])


        smpl_data_world = convert_smpl_dict_to_world(smpl_data_cam, cam_extrinsic, j_root)
        # smpl_world_2 = get_smpl_vertice(smpl_data_world)
        # visualize_projection(smpl_world_2, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")
        save_path = f'../{dataset_name}/sam3d/smpl/smpl_rescaled_{index:05d}.npz'
        smpl_data_world['scale'] = np.array(1.0, dtype=np.float32)
        np.savez(save_path, **smpl_data_world)

