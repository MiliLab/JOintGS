import numpy as np
import open3d as o3d
import cv2

def get_intrinsic_matrix(camera_dict):
    """
    将相机字典转换为 3x3 内参矩阵 K
    K = [[fx, 0,  cx],
         [0,  fy, cy],
         [0,  0,  1 ]]
    """
    model = camera_dict['model']
    params = camera_dict['params']

    # 初始化矩阵
    K = np.eye(3, dtype=np.float32)

    if model == "SIMPLE_RADIAL" or model == "SIMPLE_PINHOLE":
        # params: [f, cx, cy, (k)]
        f = params[0]
        cx = params[1]
        cy = params[2]
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy

    elif model == "PINHOLE":
        # params: [fx, fy, cx, cy]
        K[0, 0] = params[0]
        K[1, 1] = params[1]
        K[0, 2] = params[2]
        K[1, 2] = params[3]

    elif model == "RADIAL":
        # params: [f, cx, cy, k1, k2]
        f = params[0]
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = params[1]
        K[1, 2] = params[2]

    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")

    return K


def get_extrinsic_matrix(image_dict):
    """
    将包含 q 和 t 的字典转换为 4x4 外参矩阵 (World-to-Camera)
    参数:
        image_dict: 包含 'q' (qw, qx, qy, qz) 和 't' (tx, ty, tz) 的字典
    返回:
        w2c: 4x4 的 numpy 矩阵
    """
    q = image_dict['q']
    t = image_dict['t']

    # 提取四元数分量
    qw, qx, qy, qz = q

    # 1. 根据标准公式计算 3x3 旋转矩阵 R
    # 这是将四元数转换为旋转矩阵的正规数学表达
    R = np.eye(3, dtype=np.float32)
    R[0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    R[0, 1] = 2 * (qx * qy - qz * qw)
    R[0, 2] = 2 * (qx * qz + qy * qw)

    R[1, 0] = 2 * (qx * qy + qz * qw)
    R[1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    R[1, 2] = 2 * (qy * qz - qx * qw)

    R[2, 0] = 2 * (qx * qz - qy * qw)
    R[2, 1] = 2 * (qy * qz + qx * qw)
    R[2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    # 2. 构造 4x4 齐次矩阵
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    return w2c


def transform_to_world_space(points, cam_extrinsic):
    """
    将相机坐标系下的点变换到世界坐标系
    参数:
        points: (n, 3) 的 numpy 数组，表示在相机坐标系下的 SMPL 顶点
        cam_extrinsic: 4x4 的外参矩阵 (World-to-Camera, W2C)
    返回:
        points_world: (n, 3) 变换后的世界坐标点
    """
    # 1. 计算相机到世界的变换矩阵 (C2W = inv(W2C))
    c2w = np.linalg.inv(cam_extrinsic)

    # 2. 转换点云为齐次坐标 [n, 4]
    n = points.shape[0]
    points_homo = np.hstack([points, np.ones((n, 1))])

    # 3. 应用变换: P_world = C2W * P_camera
    # 注意矩阵乘法的维度，[4, 4] @ [4, n] -> [4, n]，最后转置回 [n, 4]
    points_world_homo = (c2w @ points_homo.T).T

    # 4. 返回前三列 [X, Y, Z]
    return points_world_homo[:, :3].astype(np.float32)


def auto_align_to_floor(verts_cam_6d, scene_6d, w2c):
    """
    通过交互式 Open3D 窗口手动微调缩放因子 s。
    """
    # 1. 准备基础数据
    verts_cam = verts_cam_6d[:, :3]
    pts_world = scene_6d[:, :3]
    pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]

    # 2. 初始平面拟合（作为起点）
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(pts_cam)
    plane_model, inliers = pcd_cam.segment_plane(distance_threshold=1.0,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    [a, b, c, d] = plane_model
    n = np.array([a, b, c])

    # 计算初始 s (取第1百分位数并略微缩放作为保守起始值)
    dots = verts_cam @ n
    s_candidates = -d / dots
    pos_s = s_candidates[s_candidates > 0]
    initial_s = np.percentile(pos_s, 1) * 0.95 if len(pos_s) > 0 else 1.0

    # 3. 定义交互状态
    # 使用字典存储 s，以便在闭包回调函数中修改
    state = {'s': initial_s}

    # 4. 初始化几何体
    # 场景点云
    vis_pcd = o3d.geometry.PointCloud(pcd_cam)
    vis_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    inlier_cloud = vis_pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 地面为红色

    # 人体点云
    smpl_pcd = o3d.geometry.PointCloud()
    smpl_pcd.points = o3d.utility.Vector3dVector(verts_cam * state['s'])
    smpl_pcd.paint_uniform_color([0, 0.5, 1.0])  # 人体为蓝色

    # 坐标轴
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # 5. 定义回调函数
    def update_render(vis):
        # 更新顶点坐标
        new_points = verts_cam * state['s']
        smpl_pcd.points = o3d.utility.Vector3dVector(new_points)
        # 告知渲染器几何体已更新
        vis.update_geometry(smpl_pcd)
        print(f"\r[交互中] 当前 s = {state['s']:.4f}  (方向键调整，关闭窗口确认)", end="")

    def increase_s_fine(vis):
        state['s'] += 0.05
        update_render(vis)

    def decrease_s_fine(vis):
        state['s'] -= 0.05
        update_render(vis)

    def increase_s_coarse(vis):
        state['s'] += 0.5
        update_render(vis)

    def decrease_s_coarse(vis):
        state['s'] -= 0.5
        update_render(vis)

    # 6. 启动交互式可视化
    print("\n" + "=" * 50)
    print("Open3D 交互式对齐开启：")
    print("  ↑ / ↓ : 微调 s (±0.05)")
    print("  → / ← : 粗调 s (±0.5)")
    print("  ESC   : 确认当前位置并退出")
    print("=" * 50)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Manual S-Alignment Tool", width=1200, height=800)

    # 添加几何体
    vis.add_geometry(vis_pcd)
    vis.add_geometry(smpl_pcd)
    vis.add_geometry(axes)

    # 注册按键 (GLFW key codes: 265=Up, 264=Down, 262=Right, 263=Left)
    vis.register_key_callback(265, increase_s_fine)
    vis.register_key_callback(264, decrease_s_fine)
    vis.register_key_callback(262, increase_s_coarse)
    vis.register_key_callback(263, decrease_s_coarse)

    vis.run()
    vis.destroy_window()

    print(f"\n最终选定的缩放因子 s = {state['s']:.4f}")
    return state['s']


def convert_smpl_dict_to_world(smpl_dict, cam_extrinsic, J_root_template):
    """
    J_root_template: 模板姿态下的根关节坐标 (shape: (3,) 或 (1,3))
    可以通过 smpl_model(betas=smpl_dict['betas']).joints[0, 0] 获得
    """
    world_dict = smpl_dict.copy()

    # 1. 提取相机外参 (World to Camera)
    R_w2c = cam_extrinsic[:3, :3]
    t_w2c = cam_extrinsic[:3, 3]

    # 计算 Camera to World
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    # 2. 变换 global_orient (保持不变)
    R_cam, _ = cv2.Rodrigues(smpl_dict['global_orient'].astype(np.float32))
    R_world = R_c2w @ R_cam
    global_orient_world, _ = cv2.Rodrigues(R_world)
    world_dict['global_orient'] = global_orient_world.squeeze().astype(np.float32)

    # 3. 变换 transl (补偿旋转中心偏移)
    t_cam = smpl_dict['transl'].squeeze().astype(np.float32)
    j0 = J_root_template.squeeze().astype(np.float32)

    # 修正公式：
    # t_world = R_c2w @ t_cam + t_c2w + (R_c2w @ j0 - j0)
    # 整理后：
    t_world = R_c2w @ (t_cam + j0) + t_c2w - j0

    world_dict['transl'] = t_world.astype(np.float32)

    return world_dict


import torch
from smplx import SMPL
smpl_model = SMPL(model_path="../SMPL_NEUTRAL.pkl", gender='neutral')
def get_smpl_vertice(smpl_data):
    smpl_vertice = smpl_model(betas=torch.from_numpy(smpl_data['betas'])[None], body_pose=torch.from_numpy(smpl_data['body_pose'])[None],
                                  global_orient=torch.from_numpy(smpl_data['global_orient'])[None],
                                  transl=torch.from_numpy(smpl_data['transl'])[None]).vertices.detach().cpu().numpy().squeeze()
    scale = smpl_data.get('scale') if 'scale' in smpl_data else smpl_data.get('scales')
    if scale is not None:
        smpl_vertice *= scale

    return smpl_vertice