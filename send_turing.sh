#!/bin/bash
#SBATCH -N 1                   # 
#SBATCH -n 8                   # 
#SBATCH --mem=8g               # 
#SBATCH -J "NeRF"   # 
#SBATCH -p short               # 
#SBATCH -t 12:00:00            # 
#SBATCH --gres=gpu:1           # 
#SBATCH -C "H100"         # 

#SBATCH --output turing_logs/nerf_outputs-%j.out    # Standard Output file
#SBATCH --error turing_logs/nerf_errors-%j.err    # Standard Error file

module load python             # 
module load cuda          # 

source .venv/bin/activate

python3 Phase2/train.py --data_path ./Phase2/nerf_synthetic/lego/ --n_rays_batch 16384      #
