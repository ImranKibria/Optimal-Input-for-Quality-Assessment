'''This file trains CNN model for MOS prediction with feature vectors from audio spectrogram'''

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers
import tensorflow_datasets as tfds

random.seed(999)
BATCH_SIZE = 5000

print(tf.config.list_physical_devices('GPU'))


# LOAD TRAINING & VALIDATION DATA
dataset_name = f'spectrogram_feature_dataset/default'
(ds_train_sim, ds_train_live, ds_val_sim, ds_val_live), dataset_info = tfds.load(dataset_name, 
    split=['NISQA_TRAIN_SIM','NISQA_TRAIN_LIVE', 'NISQA_VAL_SIM', 'NISQA_VAL_LIVE'], shuffle_files=True, with_info=True,)

tr_dataset = tf.data.Dataset.concatenate(ds_train_sim, ds_train_live)
val_dataset = tf.data.Dataset.concatenate(ds_val_sim, ds_val_live)

# normalization_layer = layers.Normalization(axis=(1, 2))
# normalization_layer.adapt(tr_dataset.map(lambda x: x['feature_vector']))


def unpack_data(record):
  ftr_vec = record['feature_vector']
  mos = tf.reshape(record['label'], [-1])
  return ftr_vec, mos


tr_dataset = tr_dataset.shuffle(BATCH_SIZE).map(unpack_data).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.shuffle(BATCH_SIZE).map(unpack_data).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


# CREATE & COMPLILE MODEL 
strategy = tf.distribute.MirroredStrategy()
print('Number of Devices = {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
        
  model = models.Sequential()

  model.add(layers.Conv2D(filters=16, kernel_size=(2,2), input_shape=(25, 58, 2)))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2)))

  model.add(layers.Conv2D(filters=32, kernel_size=(2,2)))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

  model.add(layers.Conv2D(filters=64, kernel_size=(2,2)))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  model.add(layers.Conv2D(filters=32, kernel_size=(2,2)))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  # Add dense layers on top
  model.add(layers.Flatten())

  model.add(layers.Dense(units=128, activation='relu'))
  model.add(layers.Dropout(rate=0.5))

  model.add(layers.Dense(units=128, activation='relu'))
  model.add(layers.Dropout(rate=0.5))

  model.add(layers.Dense(units=1, activation='relu'))
  print(model.summary())

  model.compile(optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.00004),  
              loss=tf.keras.losses.MeanSquaredError())


# DEFINE CALLBACKS
checkpoint_path = 'checkpoints/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_path, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)


# TRAIN & EVALUATE THE MODEL
EPOCHS = 500
history = model.fit(tr_dataset, epochs=EPOCHS, callbacks=[cp_callback], 
                    initial_epoch=0, validation_data=val_dataset)


# PLOT TRAINING CURVE
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.savefig('training.png')