#!/bin/bash
#SBATCH --job-name=MyPythonJob       # Job name, displayed in squeue
#SBATCH --output=slurm-%j.out        # Standard output file, %j will be replaced by job ID
#SBATCH --error=slurm-%j.err         # Standard error file
#SBATCH --nodes=1                    # Number of nodes requested
#SBATCH --ntasks=1                   # Number of tasks per node (usually 1 main process)
#SBATCH --cpus-per-task=16            # Number of CPU cores allocated per task
#SBATCH --mem=32G                    # Amount of memory allocated per node
#SBATCH --time=8:00:00               # Maximum job runtime (HH:MM:SS)
#SBATCH --gres=gpu:1              # Request 1 GPU (if GPU is needed)
#SBATCH --partition=batch            # Submit to the 'batch' partition (check your sinfo output)


source /home/louzihan/anaconda3/etc/profile.d/conda.sh
conda activate mhr


DATASETS=(
    "P0_08_outdoor_remove_jacket"
    "P1_16_outdoor_warmup"
    "P2_23_outdoor_hug_tree"
    "P3_32_outdoor_soccer_warmup_a"
    "P5_42_indoor_dancing"
    "P6_50_outdoor_workout"
    "P7_57_outdoor_rock_chair"
    "P8_67_outdoor_workout_stretch"
)

for DATA_NAME in "${DATASETS[@]}"; do
   python emdb_sam3db.py \
       --dataset_folder /opt/louzihan/dataset/emdb_refine/"$DATA_NAME" \
      --image_folder /opt/louzihan/dataset/emdb_refine/"$DATA_NAME"/images \
      --output_folder /opt/louzihan/dataset/emdb_refine/"$DATA_NAME"/sam3d/mhr \
      --checkpoint_path /opt/louzihan/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
      --mhr_path /opt/louzihan/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
      --detector_name sam3 \
      --fov_path /opt/louzihan/sam-3d-body/checkpoints/moge-2-vitl-normal/model.pt
done

