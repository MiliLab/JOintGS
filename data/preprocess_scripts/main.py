import os
import numpy as np
import torch
import open3d as o3d
from smplx import SMPL
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from utils.colmap import read_cameras_text, read_frames_text, read_images_text, read_points3D_text, rescale_colmap_dataset, get_scaled_smpl_dict
from utils.graphics import get_intrinsic_matrix, get_extrinsic_matrix, transform_to_world_space, auto_align_to_floor, convert_smpl_to_world
from utils.visualize import visualize_projection

smpl_model = SMPL(model_path="../SMPL_NEUTRAL.pkl", gender='neutral')


def get_smpl_vertice(smpl_data):
    smpl_vertice = smpl_model(betas=torch.from_numpy(smpl_data['betas'])[None], body_pose=torch.from_numpy(smpl_data['body_pose'])[None],
                                  global_orient=torch.from_numpy(smpl_data['global_orient'])[None],
                                  transl=torch.from_numpy(smpl_data['transl'])[None]).vertices.detach().cpu().numpy().squeeze()
    scale = smpl_data.get('scale') if 'scale' in smpl_data else smpl_data.get('scales')
    if scale is not None:
        smpl_vertice *= scale

    return smpl_vertice

# dataset_name = "P0_08_outdoor_remove_jacket"
# cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
# images = read_images_text(f'../{dataset_name}/sparse/0/images.txt')
# points3D = read_points3D_text(f'../{dataset_name}/sparse/0/points3D.txt')
#
# # image_00000
# index = 0
# image_name = images[index]['name']
# cam_extrinsic = get_extrinsic_matrix(images[index])
# smpl_data = dict(np.load(f'../{dataset_name}/sam3d/mhr/smpl_{index:05d}.npz', allow_pickle=True))
# smpl_vertice_cam = smpl_model(betas=torch.from_numpy(smpl_data['betas'])[None], body_pose=torch.from_numpy(smpl_data['body_pose'])[None],global_orient=torch.from_numpy(smpl_data['global_orient'])[None], transl=torch.from_numpy(smpl_data['transl'])[None]).vertices.detach().cpu().numpy().squeeze()
# smpl_vertice_world = transform_to_world_space(smpl_vertice_cam, cam_extrinsic)
# smpl_vertice_cam = np.hstack([smpl_vertice_cam, np.full_like(smpl_vertice_cam, 128)])
# smpl_vertice_world = np.hstack([smpl_vertice_world, np.full_like(smpl_vertice_world, 128)])
#
# cam_intrinsic = get_intrinsic_matrix(cameras[1])
# image_path = f'../{dataset_name}/images/{image_name}'
# H, W = cameras[1]['height'], cameras[1]['width']
# visualize_projection(smpl_vertice_world, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")
#
# s = auto_align_to_floor(smpl_vertice_cam, points3D, cam_extrinsic)
# rescale_colmap_dataset(f'../{dataset_name}/sparse/0/images.txt', f'../{dataset_name}/sparse/0/points3D.txt', s)


dataset_name = "P0_08_outdoor_remove_jacket"
cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
images = read_images_text(f'../{dataset_name}/sparse/0/images_rescaled.txt')
points3D = read_points3D_text(f'../{dataset_name}/sparse/0/points3D_rescaled.txt')
cam_intrinsic = get_intrinsic_matrix(cameras[1])


# image_00000
files_num = len([f for f in os.listdir(f"../{dataset_name}/images") if f.endswith(('.jpg', '.png'))])
files_num = 1

for i in range(files_num):
    i = 114

    image_name = images[i]['name']
    image_path = f'../{dataset_name}/images/{image_name}'
    H, W = cameras[1]['height'], cameras[1]['width']
    cam_extrinsic = get_extrinsic_matrix(images[i])
    smpl_data = dict(np.load(f'../{dataset_name}/sam3d/mhr/smpl_{i:05d}.npz', allow_pickle=True))
    smpl_vertice_cam = get_smpl_vertice(smpl_data)

    # smpl_world_1 = transform_to_world_space(smpl_vertice_cam, cam_extrinsic)
    # visualize_projection(smpl_world_1, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")

    s = auto_align_to_floor(smpl_vertice_cam, points3D, cam_extrinsic)
    if s<0.75 or s>1.25:
        s=1
    scaled_smpl_data = get_scaled_smpl_dict(smpl_data, s)


    # scaled_smpl_vertice_cam = get_smpl_vertice(scaled_smpl_data)
    # smpl_world_2 = transform_to_world_space(scaled_smpl_vertice_cam, cam_extrinsic)
    # visualize_projection(smpl_world_2, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")



    scaled_smpl_data_world = convert_smpl_to_world(scaled_smpl_data, cam_extrinsic)

    smpl_cam_3 = get_smpl_vertice(scaled_smpl_data)
    smpl_world_3 = transform_to_world_space(smpl_cam_3, cam_extrinsic)
    visualize_projection(smpl_world_3, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")

    smpl_world_4 = get_smpl_vertice(scaled_smpl_data_world)
    visualize_projection(smpl_world_4, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")

    save_path = f'../{dataset_name}/sam3d/mhr/smpl_rescaled_{i:05d}.npz'
    # np.savez(save_path, **scaled_smpl_data_world)

