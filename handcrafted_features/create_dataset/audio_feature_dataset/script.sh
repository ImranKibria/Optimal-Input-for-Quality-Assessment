#!/bin/bash

#SBATCH --time=24:00:00 
#SBATCH --job-name=ds_nisqa_create
#SBATCH --account=PAS2301
#SBATCH --mem=64gb

#SBATCH --cpus-per-task=32
#SBATCH -o /users/PAS2301/kibria5/Research/ManualFeatures_BatchNormalization/create_dataset/audio_feature_dataset/dataset.out

module load miniconda3
source activate local
cd /users/PAS2301/kibria5/Research/ManualFeatures_BatchNormalization/create_dataset/audio_feature_dataset
tfds build --overwrite