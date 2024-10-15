'''This file tests CNN model on a given testset'''

import warnings
import pandas as pd
from train_model import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import tensorflow_datasets as tfds

warnings.filterwarnings("ignore")
BATCH_SIZE = 5000 


# INPUT PIPELINE
dataset_name = f'waveform_dataset/default'
corpus_path = '/fs/ess/PAS2301/Data/Speech/NISQA_Corpus/'

(ds_val_sim, ds_val_live), dataset_info = tfds.load(dataset_name, split=['NISQA_VAL_SIM', 'NISQA_VAL_LIVE'], 
  shuffle_files=True, with_info=True, data_dir='/fs/scratch/PAS2301/Imran/')

def unpack_data(record):
  ftr_vec = record['feature_vector']
  mos = tf.reshape(record['label'], [-1])
  return ftr_vec, mos

# TODO: decide which dataset to process
test_dset = ds_val_sim.map(unpack_data).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE) 

df_val_sim = pd.read_csv(corpus_path + 'NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv')
df_val_live = pd.read_csv(corpus_path + 'NISQA_VAL_LIVE/NISQA_VAL_LIVE_file.csv')
test_df = df_val_sim   # TODO: decide which dataset to process


# CREATE & COMPLILE THE MODEL 
model = MOSnet()
model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.0005), 
              loss=tf.keras.losses.MeanSquaredError())

checkpoint_path = 'checkpoints/cp-0500.ckpt' # TODO: decide the best weights for model
model.load_weights(checkpoint_path)


# EVALUATE MODEL 
test_loss = model.evaluate(test_dset)
test_rmse = np.sqrt(test_loss)
print('RMSE (frame-level) = ', test_rmse)

deg_files = test_df['filename_deg'].to_list() # filenames of all deg waves
true_mos = test_df['mos'].to_numpy()  # true mos label of all deg waves
pred_mos = np.zeros(len(deg_files))   # predicted label of all deg waves

for i in range(len(deg_files)):
  print(f'file {i}/2500 processing')
  obsv_ds = ds_val_sim.filter(lambda x: x['filename_deg'] == deg_files[i]).map(lambda y: y['feature_vector'])
  
  index = 0
  num_records = sum(1 for _ in obsv_ds.take(-1))
  obsv_mat = np.zeros((num_records, 25, 512, 2), dtype='float32')
  for obsv in obsv_ds:
    obsv_mat[index] = np.float32(obsv)
    index += 1
  
  fr_pred = model(obsv_mat)
  pred_mos[i] = np.average(fr_pred)

rmse = (np.mean((true_mos-pred_mos)**2))**0.5
print('RMSE (utterance-level) = ', rmse)
p_corr, _ = pearsonr(true_mos, pred_mos)
print('Pearson Correlation Coefficient = ', p_corr)
s_corr, _ = spearmanr(true_mos, pred_mos)
print('Spearman Rank Correlation Coefficient = ', s_corr)

# SCATTER PLOT
plt.scatter(pred_mos, true_mos)
plt.title('NISQA_VAL_SIM')
plt.xlim([1,5]); plt.xlabel('Predicted')
plt.ylim([1,5]); plt.ylabel('True MOS')
plt.savefig('results/NISQA_VAL_SIM/scatter_plot.png')