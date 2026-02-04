import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import open3d as o3d
import os
import cv2
import pickle


def visualize_projection(points, h, w, cam_intrinsic, cam_extrinsic, if_3d=False, image_path=None, pkl_path="../SMPL_NEUTRAL.pkl"):
    """
    SMPL 人体 Mesh 可视化工具 (专用于纯人体顶点输入)
    参数:
        points: (n, 3) 或 (n, 6) 数组，仅包含人体顶点坐标
        pkl_path: 指向 SMPL .pkl 文件的路径，用于提取 faces
    """
    # 1. 提取坐标 (只取前 3 列 XYZ)
    pts_xyz = points[:, :3]

    # 定义内置的科研感蓝色
    color_human = np.array([0.65, 0.75, 0.85])

    # 2. 读取 SMPL 面片信息 (用于构建 Mesh)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        smpl_faces = np.array(data['f'], dtype=np.int32)
    else:
        raise FileNotFoundError(f"未找到 SMPL 模型文件: {pkl_path}")

    if not if_3d:
        # =========================================================
        # 模式 1: 2D 投影渲染 (基于面片的平铺渲染)
        # =========================================================
        # 世界坐标 -> 相机坐标
        points_homo = np.hstack([pts_xyz, np.ones((pts_xyz.shape[0], 1))])
        points_cam = (cam_extrinsic @ points_homo.T).T[:, :3]

        # 相机坐标 -> 像素坐标 (投影公式: u = fx*X/Z + cx)
        pts_2d_homo = (cam_intrinsic @ points_cam.T).T
        u = pts_2d_homo[:, 0] / (pts_2d_homo[:, 2] + 1e-8)
        v = pts_2d_homo[:, 1] / (pts_2d_homo[:, 2] + 1e-8)
        uv = np.stack([u, v], axis=1)

        fig, ax = plt.subplots(figsize=(10, 10 * h / w))

        # 背景图处理
        bg_img = None
        if image_path and os.path.exists(image_path):
            tmp = cv2.imread(image_path)
            if tmp is not None:
                bg_img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
                if bg_img.shape[:2] != (h, w): bg_img = cv2.resize(bg_img, (w, h))
        ax.imshow(bg_img if bg_img is not None else np.zeros((h, w, 3)))

        # --- 画家算法渲染 Mesh ---
        # 1. 获取三角形顶点的像素坐标
        tri_uvs = uv[smpl_faces]  # 形状: (Faces, 3, 2)

        # 2. 计算每个三角形的平均深度用于排序
        tri_depths = points_cam[:, 2][smpl_faces].mean(axis=1)
        sort_idx = np.argsort(tri_depths)[::-1]  # 从远到近排序

        # 3. 使用 PolyCollection 绘制实心人体
        coll = PolyCollection(tri_uvs[sort_idx],
                              facecolors=color_human,
                              edgecolors=color_human * 0.8,  # 细微边缘线增强结构
                              linewidths=0.1,
                              alpha=1.0)
        ax.add_collection(coll)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    else:
        # =========================================================
        # 模式 2: 3D 空间展示 (Open3D)
        # =========================================================
        # 创建 Mesh 对象
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(pts_xyz)
        mesh.triangles = o3d.utility.Vector3iVector(smpl_faces)

        # 着色与法线计算 (实现光影立体感)
        mesh.paint_uniform_color(color_human)
        mesh.compute_vertex_normals()

        # 可视化相机位姿
        c2w = np.linalg.inv(cam_extrinsic)
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        cam_frame.transform(c2w)

        frustum = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=w, view_height_px=h,
            intrinsic=cam_intrinsic, extrinsic=cam_extrinsic, scale=0.3
        )
        frustum.paint_uniform_color([1, 0, 0])

        o3d.visualization.draw_geometries([mesh, cam_frame, frustum],
                                          window_name="3D SMPL Pure Human Visualization")