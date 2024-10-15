#!/bin/bash

#SBATCH --time=24:00:00 
#SBATCH --job-name=ds_nisqa_create
#SBATCH --account=PAS2301
#SBATCH --mem=128gb

#SBATCH --cpus-per-task=32
#SBATCH -o /users/PAS2301/kibria5/Research/CNN_mosnet/create_dataset/spectrogram_feature_dataset/dataset.out

module load miniconda3
source activate local
cd /users/PAS2301/kibria5/Research/CNN_mosnet/create_dataset/spectrogram_feature_dataset
tfds build --overwrite
