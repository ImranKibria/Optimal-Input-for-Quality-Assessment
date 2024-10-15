#!/bin/bash

#SBATCH --time=24:00:00 
#SBATCH --job-name=ds_nisqa_create
#SBATCH --account=PAS2301
#SBATCH --mem=128gb

#SBATCH --cpus-per-task=32
#SBATCH -o /users/PAS2301/kibria5/Research/Learnable_MOSnet/create_dataset/waveform_dataset/dataset.out

module load miniconda3
source activate local
cd /users/PAS2301/kibria5/Research/Learnable_MOSnet/create_dataset/waveform_dataset
tfds build --overwrite --data_dir /fs/scratch/PAS2301/Imran