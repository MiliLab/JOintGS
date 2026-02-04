import os
import numpy as np
import torch
import open3d as o3d
from smplx import SMPL
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from utils.colmap import read_cameras_text, read_frames_text, read_images_text, read_points3D_text, rescale_colmap_dataset, get_scaled_smpl_dict
from utils.graphics import get_intrinsic_matrix, get_extrinsic_matrix, transform_to_world_space, auto_align_to_floor, convert_smpl_dict_to_world
from utils.visualize import visualize_projection


smpl_model = SMPL(model_path="../SMPL_NEUTRAL.pkl", gender='neutral')

def get_smpl_joints(smpl_data, smpl_model):
    """
    提取带缩放的 SMPL 3D 关节点坐标
    """
    # 转换 tensor
    betas = torch.from_numpy(smpl_data['betas']).float().view(1, -1)
    body_pose = torch.from_numpy(smpl_data['body_pose']).float().view(1, -1)
    global_orient = torch.from_numpy(smpl_data['global_orient']).float().view(1, -1)
    transl = torch.from_numpy(smpl_data['transl']).float().view(1, -1)

    # 前向计算获取 joints
    output = smpl_model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl
    )

    # 提取关节点 (通常是 24 或 45 个点，取决于模型)
    joints = output.joints.detach().cpu().numpy().squeeze() # (J, 3)

    # 应用 scale
    scale = smpl_data.get('scale', smpl_data.get('scales', 1.0))
    joints *= scale

    return joints

def get_smpl_vertice(smpl_data):
    smpl_vertice = smpl_model(betas=torch.from_numpy(smpl_data['betas'])[None], body_pose=torch.from_numpy(smpl_data['body_pose'])[None],
                                  global_orient=torch.from_numpy(smpl_data['global_orient'])[None],
                                  transl=torch.from_numpy(smpl_data['transl'])[None]).vertices.detach().cpu().numpy().squeeze()
    scale = smpl_data.get('scale') if 'scale' in smpl_data else smpl_data.get('scales')
    if scale is not None:
        smpl_vertice *= scale

    return smpl_vertice

def build_extrinsic_matrix(cam_data):
    # 1. 提取旋转数据
    rot = cam_data['cam_rot'] # 可能是 (3,3) 或 (3,)
    transl = cam_data['cam_transl'].reshape(3, 1) # 确保是列向量 (3,1)

    # 2. 如果旋转是向量格式 (Rodrigues)，转换为 3x3 矩阵
    if rot.size == 3:
        R, _ = cv2.Rodrigues(rot)
    else:
        R = rot

    # 3. 拼接成 4x4 矩阵
    extrinsic = np.eye(4) # 初始化为单位阵
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:4] = transl

    return extrinsic


def project_3d_to_2d(joints_3d, intrinsic, extrinsic):
    """
    将 3D 关节点投影到 2D 像素平面
    joints_3d: (J, 3)
    intrinsic: (3, 3)
    extrinsic: (4, 4)
    """
    # 1. 转换到相机坐标系 (R*X + t)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3:]
    joints_cam = (R @ joints_3d.T + t).T  # (J, 3)

    # 2. 投影到像素平面 (K * X_cam / Z_cam)
    joints_2d = (intrinsic @ joints_cam.T).T
    joints_2d = joints_2d[:, :2] / joints_2d[:, 2:] # 归一化并舍弃 Z

    return joints_2d

def calculate_2d_pixel_error(pts2d_pred, pts2d_gt):
    """计算两组 2D 点之间的平均像素距离"""
    dist = np.linalg.norm(pts2d_pred - pts2d_gt, axis=-1)
    return np.mean(dist)

index_list = [i for i in range(100)]
if_vis = False
index_list = [2]
if_vis = True
for index in index_list:
    dataset_name = "P0_08_outdoor_remove_jacket"

    images_dir = f'../{dataset_name}/images'
    image_files = sorted(os.listdir(images_dir))
    image_index = int(image_files[index][:-4])
    print(image_index)
    # cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
    # images = read_images_text(f'../{dataset_name}/sparse/0/images.txt')
    # image_name = images[index]['name']
    # cam_extrinsic = get_extrinsic_matrix(images[index])
    # smpl_data = dict(np.load(f'../{dataset_name}/sam3d/smpl/smpl_{index:05d}.npz', allow_pickle=True))
    # smpl_vertice_cam = get_smpl_vertice(smpl_data)
    # smpl_vertice_world = transform_to_world_space(smpl_vertice_cam, cam_extrinsic)
    # cam_intrinsic = get_intrinsic_matrix(cameras[1])
    # image_path = f'../{dataset_name}/images/{image_name}'
    # H, W = cameras[1]['height'], cameras[1]['width']
    # visualize_projection(smpl_vertice_world, H, W, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")


    cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
    images = read_images_text(f'../{dataset_name}/sparse/0/images_rescaled.txt')
    image_name = images[image_index]['name']
    cam_extrinsic1 = get_extrinsic_matrix(images[image_index])
    smpl_data1 = dict(np.load(f'../{dataset_name}/sam3d/smpl/smpl_rescaled_{image_index:05d}.npz', allow_pickle=True))
    smpl_vertice_world = get_smpl_vertice(smpl_data1)
    cam_intrinsic = get_intrinsic_matrix(cameras[1])
    image_path = f'../{dataset_name}/images/{image_name}'
    H, W = cameras[1]['height'], cameras[1]['width']
    if if_vis:
        visualize_projection(smpl_vertice_world, H, W, cam_intrinsic, cam_extrinsic1, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")


    cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
    images = read_images_text(f'../{dataset_name}/sparse/0/images_rescaled.txt')
    image_name = images[image_index]['name']
    cam_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\cam_optimized_by_model.npz', allow_pickle=True))
    smpl_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\smpl_optimized_by_model.npz', allow_pickle=True))

    cam_data2 = {
        'cam_rot': cam_data_list['cam_rot'][index],
        'cam_transl': cam_data_list['cam_transl'][index]
    }
    cam_extrinsic2 = build_extrinsic_matrix(cam_data2)
    smpl_data2 = {
        'betas':smpl_data_list['betas'][index],
        'body_pose':smpl_data_list['body_pose'][index],
        'global_orient':smpl_data_list['global_orient'][index],
        'transl':smpl_data_list['transl'][index],
        'scale':smpl_data_list['scale'][index]

    }
    smpl_vertice_world = get_smpl_vertice(smpl_data2)
    cam_intrinsic = get_intrinsic_matrix(cameras[1])
    image_path = f'../{dataset_name}/images/{image_name}'
    H, W = cameras[1]['height'], cameras[1]['width']
    if if_vis:
        visualize_projection(smpl_vertice_world, H, W, cam_intrinsic, cam_extrinsic2, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")


    cameras = read_cameras_text(f'../{dataset_name}/sparse/0/cameras.txt')
    images = read_images_text(f'../{dataset_name}/sparse/0/images_rescaled.txt')
    image_name = images[image_index]['name']
    cam_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\cam_optimized_by_model.npz', allow_pickle=True))
    smpl_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\origin_smpl.npz', allow_pickle=True))
    cam_data3 = {
        'cam_rot': cam_data_list['cam_rot'][index],
        'cam_transl': cam_data_list['cam_transl'][index]
    }
    cam_extrinsic3 = build_extrinsic_matrix(cam_data3)
    smpl_data3 = {
        'betas':smpl_data_list['betas'][index],
        'body_pose':smpl_data_list['body_pose'][index],
        'global_orient':smpl_data_list['global_orient'][index],
        'transl':smpl_data_list['transl'][index],
        'scale':smpl_data_list['scale'][index]

    }
    smpl_vertice_world = get_smpl_vertice(smpl_data3)
    cam_intrinsic = get_intrinsic_matrix(cameras[1])
    image_path = f'../{dataset_name}/images/{image_name}'
    H, W = cameras[1]['height'], cameras[1]['width']
    if if_vis:
        visualize_projection(smpl_vertice_world, H, W, cam_intrinsic, cam_extrinsic3, if_3d=False, image_path=image_path, pkl_path="../SMPL_NEUTRAL.pkl")


    joints3d_gt = get_smpl_joints(smpl_data1, smpl_model)
    joints2d_gt = project_3d_to_2d(joints3d_gt, cam_intrinsic, cam_extrinsic1)

    # --- 第二组: Optimized ---
    joints3d_opt = get_smpl_joints(smpl_data2, smpl_model)
    joints2d_opt = project_3d_to_2d(joints3d_opt, cam_intrinsic, cam_extrinsic2)
    error_2d_opt = calculate_2d_pixel_error(joints2d_opt, joints2d_gt)

    # --- 第三组: Origin ---
    joints3d_ori = get_smpl_joints(smpl_data3, smpl_model)
    joints2d_ori = project_3d_to_2d(joints3d_ori, cam_intrinsic, cam_extrinsic3)
    error_2d_ori = calculate_2d_pixel_error(joints2d_ori, joints2d_gt)

    print(f"{index}: {error_2d_ori-error_2d_opt}")


