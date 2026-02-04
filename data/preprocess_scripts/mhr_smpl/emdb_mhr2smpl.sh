#!/bin/bash
#SBATCH --job-name=MyPythonJob       # Job name, displayed in squeue
#SBATCH --output=slurm-%j.out        # Standard output file, %j will be replaced by job ID
#SBATCH --error=slurm-%j.err         # Standard error file
#SBATCH --nodes=1                    # Number of nodes requested
#SBATCH --ntasks=1                   # Number of tasks per node (usually 1 main process)
#SBATCH --cpus-per-task=16            # Number of CPU cores allocated per task
#SBATCH --mem=32G                    # Amount of memory allocated per node
#SBATCH --time=0:20:00               # Maximum job runtime (HH:MM:SS)
#SBATCH --gres=gpu:1              # Request 1 GPU (if GPU is needed)
#SBATCH --partition=batch            # Submit to the 'batch' partition (check your sinfo output)


source /home/louzihan/anaconda3/etc/profile.d/conda.sh
conda activate mhr3

python emdb_mhr2smpl.py
