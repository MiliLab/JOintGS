# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
import struct
from glob import glob

import pyrootutils
import cv2
import numpy as np
import torch
from tqdm import tqdm

# 假设这些是从您的项目包中导入的
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

def recursive_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        return {k: recursive_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_numpy(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_numpy(x) for x in obj)
    else:
        return obj

def read_intrinsics_from_colmap(txt_path, device="cuda"):
    """
    解析 COLMAP cameras.txt 文件并返回 (1, 3, 3) 的内参矩阵
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到文件: {txt_path}")

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            elems = line.split()
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = [float(x) for x in elems[4:]]

            if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")

            intrinsics = np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ], dtype=np.float32)

            cam_intrinsics = torch.from_numpy(intrinsics).unsqueeze(0).to(device)
            return cam_intrinsics, width, height
    
    raise ValueError(f"No valid camera found in {txt_path}")

def main(args):
    # 环境路径获取
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # 初始化模型
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector
        human_detector = HumanDetector(name=args.detector_name, device=device, path=detector_path)
    
    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(name=args.segmentor_name, device=device, path=segmentor_path)
        
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    # 读取内参 (注意解包 read_intrinsics_from_colmap 的三个返回值)
    # colmap_path = os.path.join(args.dataset_folder, 'sparse/0/cameras.txt')
    # cam_intrinsics, img_w, img_h = read_intrinsics_from_colmap(colmap_path, device=device)

    # 路径动态设置
    dataset_name = os.path.basename(args.dataset_folder.strip('/'))
    # 如果命令行没给 image_folder，则根据 dataset_folder 自动推导
    if not args.image_folder:
        args.image_folder = os.path.join(args.dataset_folder, 'images')
    if not args.output_folder:
        args.output_folder = os.path.join(args.dataset_folder, 'sam3d')
    
    os.makedirs(args.output_folder, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    images_list = sorted([
        img for ext in image_extensions 
        for img in glob(os.path.join(args.image_folder, ext))
    ])

    print(f"Processing {len(images_list)} images from: {args.image_folder}")

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            # cam_int=cam_intrinsics,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )
        
        save_name_base = os.path.splitext(os.path.basename(image_path))[0]
	
        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        vis_save_path = os.path.join(args.output_folder, 'images', f"{save_name_base}.jpg")
        cv2.imwrite(vis_save_path, rend_img.astype(np.uint8))



        param_save_path = os.path.join(args.output_folder, 'mhr', f"{save_name_base}.npz")
        clean_outputs = recursive_to_numpy(outputs)
        np.savez(param_save_path, outputs=clean_outputs, image_path=image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM 3D Body Demo")
    
    # 新加的必填参数
    parser.add_argument("--dataset_folder", required=True, type=str, help="COLMAP dataset root")
    
    parser.add_argument("--image_folder", type=str, default="", help="Input images folder")
    parser.add_argument("--output_folder", default="", type=str, help="Output folder")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Model checkpoint")
    
    parser.add_argument("--detector_name", default="vitdet", type=str)
    parser.add_argument("--segmentor_name", default="sam2", type=str)
    parser.add_argument("--fov_name", default="moge2", type=str)
    
    parser.add_argument("--detector_path", default="", type=str)
    parser.add_argument("--segmentor_path", default="", type=str)
    parser.add_argument("--fov_path", default="", type=str)
    parser.add_argument("--mhr_path", default="", type=str)
    
    parser.add_argument("--bbox_thresh", default=0.8, type=float)
    parser.add_argument("--use_mask", action="store_true", default=True)

    args = parser.parse_args()
    main(args)