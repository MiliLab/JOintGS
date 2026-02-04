import numpy as np
import open3d as o3d
import os


def read_cameras_text(path):
    """
    读取 cameras.txt 文件
    返回格式: {camera_id: {'model': str, 'width': int, 'height': int, 'params': np.array}}
    """
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array(parts[4:], dtype=np.float32)

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras


def read_images_text(path):
    """
    读取 images.txt 文件并按文件名序号排序
    返回格式: {frame_idx: {'q': np.array, 't': np.array, 'camera_id': int, 'name': str, 'image_id': int}}
    """
    images_raw = {}
    with open(path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            parts = line.split()
            # 1. 提取原始数据
            image_id = int(parts[0])
            qvec = np.array(parts[1:5], dtype=np.float32)
            tvec = np.array(parts[5:8], dtype=np.float32)
            camera_id = int(parts[8])
            image_name = parts[9]

            # 2. 从文件名提取序号 (例如 '00005.jpg' -> 5)
            # os.path.splitext 分割文件名和后缀，int() 自动处理前导零
            try:
                frame_idx = int(os.path.splitext(image_name)[0])
            except ValueError:
                # 如果文件名不是纯数字（如 'frame_001.jpg'），则需要更复杂的正则，这里先按纯数字处理
                frame_idx = image_id

            images_raw[frame_idx] = {
                "q": qvec,
                "t": tvec,
                "camera_id": camera_id,
                "name": image_name,
                "image_id": image_id  # 保留原始 ID 备用
            }

            i += 2  # 跳过 2D 点那一行

    # 3. 按键（帧序号）进行排序，返回一个有序字典
    sorted_keys = sorted(images_raw.keys())
    images_sorted = {k: images_raw[k] for k in sorted_keys}

    return images_sorted


def read_frames_text(path):
    """
    读取 frames.txt 文件
    返回格式: {frame_id: {'rig_id': int, 'q': np.array, 't': np.array, 'sensor_info': list}}
    """
    frames = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()

            # 格式: FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ, ...
            frame_id = int(parts[0])
            rig_id = int(parts[1])
            qvec = np.array(parts[2:6], dtype=np.float32)
            tvec = np.array(parts[6:9], dtype=np.float32)

            # 剩余部分通常是传感器元数据 (SENSOR_TYPE, SENSOR_ID, DATA_ID)
            sensor_info = parts[9:]

            frames[frame_id] = {
                "rig_id": rig_id,
                "q": qvec,
                "t": tvec,
                "sensor_info": sensor_info
            }
    return frames


def read_points3D_text(path):
    """
    读取 COLMAP points3D.txt 文件
    1. 提取 XYZ 和 RGB 数据
    2. 保存为同名 .ply 文件
    3. 返回 (n, 6) 的 numpy 数组 [x, y, z, r, g, b]
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        # =========================================================
        # 逻辑 1: 处理 COLMAP .txt 文件
        # =========================================================
        points, colors = [], []

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # 提取坐标 [1:4] 和 颜色 [4:7]
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([int(parts[4]), int(parts[5]), int(parts[6])])

        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)

        # 自动保存为 .ply 备份
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        ply_path = os.path.splitext(path)[0] + ".ply"
        o3d.io.write_point_cloud(ply_path, pcd)

    elif ext == ".ply":
        # =========================================================
        # 逻辑 2: 直接读取 .ply 文件
        # =========================================================
        pcd = o3d.io.read_point_cloud(path)

        points = np.asarray(pcd.points, dtype=np.float32)
        # Open3D colors 是 [0, 1] 浮点数，需要转回 [0, 255]
        colors = (np.asarray(pcd.colors, dtype=np.float32) * 255.0)

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # 合并为 (n, 6)
    points_6d = np.hstack([points, colors])
    return points_6d


def rescale_colmap_dataset(images_path, points3d_path, s):
    """
    根据缩放因子 s 重新缩放 COLMAP 数据集
    参数:
        images_path: images.txt 的路径
        points3d_path: points3D.txt 的路径
        s: 缩放因子 (Human_Scale * s = Scene_Scale)
    """
    # 1. 处理 points3D.txt
    rescaled_pts_path = os.path.join(os.path.dirname(points3d_path), "points3D_rescaled.txt")

    with open(points3d_path, "r") as f_in, open(rescaled_pts_path, "w") as f_out:
        for line in f_in:
            if line.startswith("#") or not line.strip():
                f_out.write(line)
                continue

            parts = line.split()
            # 提取 XYZ 并缩放
            x, y, z = float(parts[1]) / s, float(parts[2]) / s, float(parts[3]) / s
            # 重新组合行: ID, X, Y, Z, R, G, B, ERROR, TRACK...
            new_line = f"{parts[0]} {x} {y} {z} {' '.join(parts[4:])}\n"
            f_out.write(new_line)

    # 2. 处理 images.txt
    rescaled_imgs_path = os.path.join(os.path.dirname(images_path), "images_rescaled.txt")

    with open(images_path, "r") as f_in, open(rescaled_imgs_path, "w") as f_out:
        lines = f_in.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("#") or not line.strip():
                f_out.write(line)
                i += 1
                continue

            # 第一行: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            parts = line.split()
            tx, ty, tz = float(parts[5]) / s, float(parts[6]) / s, float(parts[7]) / s

            # 重新构造第一行
            new_line = f"{parts[0]} {' '.join(parts[1:5])} {tx} {ty} {tz} {' '.join(parts[8:])}\n"
            f_out.write(new_line)

            # 第二行 (2D points) 直接原样复制
            if i + 1 < len(lines):
                f_out.write(lines[i + 1])

            i += 2


def get_scaled_smpl_dict(smpl_data, s):
    """
    计算缩放后的 SMPL 数据字典，不进行磁盘保存
    参数:
        smpl_data: 原始从 .npz 加载的 dict
        s: 缩放因子 (float)
    返回:
        updated_data: 更新了 transl 和 scale 键的字典
    """
    # 1. 浅拷贝字典，避免修改原始输入数据
    updated_data = smpl_data.copy()

    # 2. 同步缩放平移向量 (Translation)
    # 核心原理：在相机空间中，将人体放大 s 倍的同时推远 s 倍
    # 这样在投影公式 f * (s*X / s*Z) 中，s 被抵消，2D 对齐效果保持完美
    # if 'transl' in updated_data:
    #     updated_data['transl'] = (updated_data['transl'] * s).astype(np.float32)

    # 3. 记录缩放因子，方便后续追溯或反向计算
    updated_data['scale'] = np.array(s, dtype=np.float32)

    return updated_data
