import os
import numpy as np
import torch
import smplx
import cv2
from tqdm import tqdm
from mhr.mhr import MHR
from tools.mhr_smpl_conversion.conversion import Conversion


def process_and_save_batch(vertices_list, cam_t_list, paths_list):
    if not vertices_list:
        return
    batch_tensor = torch.from_numpy(np.stack(vertices_list)).float().to(device) * 100

    batch_results = converter.convert_mhr2smpl(
        mhr_vertices=batch_tensor,
        return_smpl_parameters=True,
        return_smpl_meshes=True
    )

    params_dict = batch_results.result_parameters
    errors_array = batch_results.result_errors

    for i in range(len(paths_list)):
        original_npz_path = paths_list[i]
        save_path_npz = os.path.join(os.path.dirname(original_npz_path), "smpl_" + os.path.basename(original_npz_path))

        single_result = {}
        if params_dict is not None:
            for key, value in params_dict.items():
                val = value[i].detach().cpu().numpy() if isinstance(value, torch.Tensor) else value[i]
                # 将 pred_cam_t 补偿到 SMPL 的 transl 参数中
                if key == 'transl':
                    val = val + cam_t_list[i]
                single_result[key] = val


        np.savez(save_path_npz, **single_result)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mhr_model = MHR.from_files(lod=1, device=torch.device("cpu"))
smpl_model = smplx.SMPL(model_path="tools/mhr_smpl_conversion/assets/SMPL_NEUTRAL.pkl", gender='neutral')
converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model, method="pytorch")

BATCH_SIZE = 8
batch_vertices_list, batch_cam_t_list, batch_paths_list = [], [], []

dataset_name = "P0_08_outdoor_remove_jacket"
image_dir = f"../dataset/emdb_refine/{dataset_name}/images"
mhr_dir = f"../dataset/emdb_refine/{dataset_name}/sam3d/mhr"

files = os.listdir(image_dir)
for i in tqdm(range(len(files))):
    idx = f'{i:05d}'
    mhr_path = os.path.join(mhr_dir, f"{idx}.npz")
    image_path = os.path.join(image_dir, f"{idx}.jpg")
    if not os.path.exists(mhr_path): continue

    data = np.load(mhr_path, allow_pickle=True)
    mhr_results = data['outputs'][()]
    if len(mhr_results) > 0:
        mhr_dict = max(mhr_results, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))
    else:
        mhr_dict = mhr_results[0] if isinstance(mhr_results, list) else mhr_results

    img = cv2.imread(image_path)
    if img is not None:
        bx = mhr_dict['bbox'].astype(int)
        cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(mhr_dir, f"smpl_{idx}.jpg"), img)

    batch_vertices_list.append(mhr_dict['pred_vertices'])
    batch_cam_t_list.append(mhr_dict['pred_cam_t'].flatten())
    batch_paths_list.append(mhr_path)

    if len(batch_vertices_list) == BATCH_SIZE:
        process_and_save_batch(batch_vertices_list, batch_cam_t_list, batch_paths_list)
        batch_vertices_list, batch_cam_t_list, batch_paths_list = [], [], []

if batch_vertices_list:
    process_and_save_batch(batch_vertices_list, batch_cam_t_list, batch_paths_list)