#!/bin/bash

#SBATCH --time=72:00:00 
#SBATCH --job-name=MOS-Pred
#SBATCH --account=PAS2301

#SBATCH --mem=64gb
#SBATCH -o /users/PAS2301/kibria5/Research/Learnable_MOSnet/results/NISQA_VAL_SIM/testing.out

#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1

module load miniconda3
source activate local
module load cuda/11.8.0
python test_model.py