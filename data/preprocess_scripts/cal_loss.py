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

def calculate_mpjpe(joints_pred, joints_gt):

    """

    计算 MPJPE (单位: mm)

    joints_pred: 预测的关节坐标 (J, 3)

    joints_gt: 真值的关节坐标 (J, 3)

    """

    # 计算每个关节点的欧式距离: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)

    dist = np.linalg.norm(joints_pred - joints_gt, axis=-1)



    # 取平均值

    mpjpe = np.mean(dist)



    # 假设原始单位是 m，转为 mm (如果已经是 mm 则去掉 * 1000)

    return mpjpe * 1000

def p_mpjpe(joints_pred, joints_gt):
    """
    计算 P-MPJPE: 先进行刚性对齐（平移+旋转），再计算 MPJPE
    """
    # 1. 平移对齐 (将质心移至原点)
    mu_pred = joints_pred.mean(axis=0)
    mu_gt = joints_gt.mean(axis=0)

    j_pred_centered = joints_pred - mu_pred
    j_gt_centered = joints_gt - mu_gt

    # 2. 旋转对齐 (使用 SVD 寻找最优旋转矩阵 R)
    # R, scale = orthogonal_procrustes(j_gt_centered, j_pred_centered) # 这是对齐预测到真值
    # 注意: orthogonal_procrustes 返回的 R 是满足 A = B @ R.T 的旋转
    A = j_gt_centered
    B = j_pred_centered

    M = B.T @ A
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    # 确保 R 是一个右手系的旋转矩阵（防止镜像）
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = U @ Vt

    # 应用旋转
    j_pred_aligned = B @ R

    # 3. 计算对齐后的 MPJPE (单位 mm)
    # 此时平移已经归零，旋转已经对齐
    error = np.linalg.norm(j_pred_aligned - A, axis=-1)
    return np.mean(error) * 1000

for index in range(100):
    dataset_name = "P0_08_outdoor_remove_jacket"

    images_dir = f'../{dataset_name}/images'
    image_files = sorted(os.listdir(images_dir))
    image_index = int(image_files[index][:-4])

    smpl_data1 = dict(np.load(f'../{dataset_name}/sam3d/smpl/smpl_rescaled_{image_index:05d}.npz', allow_pickle=True))

    smpl_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\smpl_optimized_by_model.npz', allow_pickle=True))
    smpl_data2 = {
        'betas':smpl_data_list['betas'][index],
        'body_pose':smpl_data_list['body_pose'][index],
        'global_orient':smpl_data_list['global_orient'][index],
        'transl':smpl_data_list['transl'][index],
        'scale':smpl_data_list['scale'][index]

    }

    smpl_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\origin_smpl.npz', allow_pickle=True))
    smpl_data3 = {
        'betas':smpl_data_list['betas'][index],
        'body_pose':smpl_data_list['body_pose'][index],
        'global_orient':smpl_data_list['global_orient'][index],
        'transl':smpl_data_list['transl'][index],
        'scale':smpl_data_list['scale'][index]
    }

    joints1 = get_smpl_joints(smpl_data1, smpl_model) # Ground Truth
    joints2 = get_smpl_joints(smpl_data2, smpl_model) # Optimized
    joints3 = get_smpl_joints(smpl_data3, smpl_model) # Origin

    error_optimized = p_mpjpe(joints2, joints1)
    error_origin = p_mpjpe(joints3, joints1)

    # print(f"Index {index}:")
    # print(f"  Optimized SMPL MPJPE: {error_optimized:.4f} mm")
    # print(f"  Original SMPL MPJPE: {error_origin:.4f} mm")
    print(f"  {index}: {error_origin - error_optimized:.4f} mm")