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

def get_camera_center(extrinsic):
    """从 4x4 外参矩阵提取相机的世界坐标中心"""
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    center = -R.T @ t
    return center

def create_trajectory_line(centers, color):
    """创建 Open3D 线条集合"""
    lines = [[i, i + 1] for i in range(len(centers) - 1)]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(centers)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

index_list = [i for i in range(100)]
traj1, traj2, traj3 = [], [], []

for index in index_list:
    dataset_name = "P0_08_outdoor_remove_jacket"

    images_dir = f'../{dataset_name}/images'
    image_files = sorted(os.listdir(images_dir))
    image_index = int(image_files[index][:-4])
    images = read_images_text(f'../{dataset_name}/sparse/0/images_rescaled.txt')

    cam_extrinsic1 = get_extrinsic_matrix(images[image_index])

    cam_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\cam_optimized_by_model.npz', allow_pickle=True))
    cam_data2 = {
        'cam_rot': cam_data_list['cam_rot'][index],
        'cam_transl': cam_data_list['cam_transl'][index]
    }
    cam_extrinsic2 = build_extrinsic_matrix(cam_data2)

    cam_data_list = dict(np.load(f'C:\\Users\\louzihan\\Desktop\\cam_optimized_by_model.npz', allow_pickle=True))
    cam_data3 = {
        'cam_rot': cam_data_list['cam_rot'][index],
        'cam_transl': cam_data_list['cam_transl'][index]
    }
    cam_extrinsic3 = build_extrinsic_matrix(cam_data3)

    traj1.append(get_camera_center(cam_extrinsic1))
    traj2.append(get_camera_center(cam_extrinsic2))
    traj3.append(get_camera_center(cam_extrinsic3))


traj1, traj2, traj3 = np.array(traj1), np.array(traj2), np.array(traj3)

# 创建三条线
# 轨迹1 (GT/Original): 红色
# 轨迹2 (Optimized): 绿色
# 轨迹3 (Third Source): 蓝色
line_gt = create_trajectory_line(traj1, [1, 0, 0])
line_opt = create_trajectory_line(traj2, [0, 1, 0])
line_ori = create_trajectory_line(traj3, [0, 0, 1])

# 可视化
print("正在展示相机轨迹: 红色(GT), 绿色(Optimized), 蓝色(Origin)")
o3d.visualization.draw_geometries([line_ori])