'''This file tests CNN model on a given testset'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import tensorflow_datasets as tfds

import warnings

warnings.filterwarnings("ignore")
dir_name = 'NISQA_VAL_SIM'
BATCH_SIZE = 5000 


# INPUT PIPELINE
dataset_name = f'spectrogram_feature_dataset/default'
(ds_train_sim, ds_train_live, ds_val_sim, ds_val_live), dataset_info = tfds.load(dataset_name, 
    split=['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE', 'NISQA_VAL_SIM', 'NISQA_VAL_LIVE'], shuffle_files=True, with_info=True,)

def unpack_data(record):
  ftr_vec = record['feature_vector']
  mos_label = tf.reshape(record['label'], [-1])
  return ftr_vec, mos_label

if dir_name == 'NISQA_VAL_SIM':
  chosen_ds = ds_val_sim
elif dir_name == 'NISQA_VAL_LIVE':
  chosen_ds = ds_val_live

test_ds = chosen_ds.map(unpack_data).batch(BATCH_SIZE)

# CREATE & COMPLILE THE MODEL 
checkpoint_path = 'checkpoints/cp-0011.ckpt' # TODO: decide the best weights for model
model = tf.keras.models.load_model(checkpoint_path)


# EVALUATE MODEL 
test_loss = model.evaluate(test_ds)
test_rmse = np.sqrt(test_loss)
print('RMSE (frame-level) = ', test_rmse)

# EVALUATE MODEL AT UTTERANCE_LEVEL
corpus_path = '/fs/ess/PAS2301/Data/Speech/NISQA_Corpus/'
if dir_name == 'NISQA_VAL_SIM':
  test_df = pd.read_csv(corpus_path + 'NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv')
elif dir_name == 'NISQA_VAL_LIVE':
  test_df = pd.read_csv(corpus_path + 'NISQA_VAL_LIVE/NISQA_VAL_LIVE_file.csv')

deg_files = test_df['filename_deg'].to_list() # filenames of all deg waves
true_mos = test_df['mos'].to_numpy()  # true mos label of all deg waves
pred_mos = np.zeros(len(deg_files))   # predicted label of all deg waves

for i in range(len(deg_files)):
  obsv_ds = chosen_ds.filter(lambda x: x['filename_deg'] == deg_files[i]).map(lambda y: tf.expand_dims(y['feature_vector'], axis=0))
  fr_pred = model.predict(obsv_ds)
  utt_mos = np.average(fr_pred)
  pred_mos[i] = utt_mos

rmse = (np.mean((true_mos-pred_mos)**2))**0.5
print('RMSE (utterance-level) = ', rmse)
p_corr, _ = pearsonr(true_mos, pred_mos)
print('Pearson Correlation Coefficient = ', p_corr)
s_corr, _ = spearmanr(true_mos, pred_mos)
print('Spearman Rank Correlation Coefficient = ', s_corr)

# SCATTER PLOT
plt.scatter(pred_mos, true_mos)
plt.title(dir_name)
plt.xlim([1,5]); plt.xlabel('Predicted')
plt.ylim([1,5]); plt.ylabel('True MOS')
plt.savefig('results/' + dir_name + '/scatter_plot.png')