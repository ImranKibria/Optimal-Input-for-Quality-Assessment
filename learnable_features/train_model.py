import os
import librosa
import numpy as np
import tensorflow as tf
from keras import models, layers
from tensorflow import keras
import tensorflow_hub as hub
from keras.models import Model

BATCH_SIZE = 5000
corpus_path = '/fs/ess/PAS2301/Data/Speech/NISQA_Corpus'


class Predictor(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_layer_1 = layers.Conv2D(filters=16, kernel_size=(2,2), input_shape=(25, 58, 2))
        self.batch_norm_1 = layers.BatchNormalization()
        self.non_linearity = layers.ReLU()
        self.max_pooling_1 = layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2))

        self.conv_layer_2 = layers.Conv2D(filters=32, kernel_size=(2,2))
        self.batch_norm_2 = layers.BatchNormalization()
        self.non_linearity = layers.ReLU()
        self.max_pooling_2 = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_layer_3 = layers.Conv2D(filters=64, kernel_size=(2,2))
        self.batch_norm_3 = layers.BatchNormalization()
        self.non_linearity = layers.ReLU()

        self.conv_layer_4 = layers.Conv2D(filters=32, kernel_size=(2,2))
        self.batch_norm_4 = layers.BatchNormalization()
        self.non_linearity = layers.ReLU()

        # Add dense layers on top
        self.flatten_layer = layers.Flatten()

        self.dense_layer_1 = layers.Dense(units=128, activation='relu')
        self.dropout_layer = layers.Dropout(rate=0.5)

        self.dense_layer_2 = layers.Dense(units=128, activation='relu')
        self.dropout_layer = layers.Dropout(rate=0.5)

        self.dense_layer_3 = layers.Dense(units=1, activation='relu')

    def call(self, input, training=False):
        x = self.conv_layer_1(input)
        x = self.batch_norm_1(x, training=training)
        x = self.non_linearity(x)
        x = self.max_pooling_1(x)

        x = self.conv_layer_2(x)
        x = self.batch_norm_2(x, training=training)
        x = self.non_linearity(x)
        x = self.max_pooling_2(x)

        x = self.conv_layer_3(x)
        x = self.batch_norm_3(x, training=training)
        x = self.non_linearity(x)

        x = self.conv_layer_4(x)
        x = self.batch_norm_4(x, training=training)
        x = self.non_linearity(x)

        # Add dense layers on top
        x = self.flatten_layer(x)

        x = self.dense_layer_1(x)
        x = self.dropout_layer(x, training=training)

        x = self.dense_layer_2(x)
        x = self.dropout_layer(x, training=training)

        output = self.dense_layer_3(x)

        return output


class Encoder(layers.Layer): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self.ref_feature_layer = layers.Dense(units=58, activation='ReLU')
        self.deg_feature_layer = layers.Dense(units=58, activation='ReLU')

    def call(self, ref_wav, deg_wav):
        '''
        Input shape: BATCH_SIZE x 25 x 512
        Output shape: BATCH_SIZE x 25 x 58 x 2
        Unique features are learned for 58 dims & 2 files.
        '''
        
        ref_wav = tf.reshape(ref_wav, [-1, 512])
        deg_wav = tf.reshape(deg_wav, [-1, 512])
        
        ref_wav = self.ref_feature_layer(ref_wav) 
        deg_wav = self.deg_feature_layer(deg_wav) 
        
        ref_wav = tf.reshape(ref_wav, [-1, 25, 58])
        deg_wav = tf.reshape(deg_wav, [-1, 25, 58])
        
        audio = tf.stack([ref_wav, deg_wav], axis=3) 
        return audio


class MOSnet(Model):
    def __init__(self, training=False):
        super(MOSnet, self).__init__()
        """
        Architecture Design Defined Here
        """
        self.encoder = Encoder()
        self.predictor = Predictor()        


    def compile(self, loss, optimizer, **kwargs): 
        super().compile(**kwargs)
        """
        Loss, Optimizers, & Metrics Defined Here
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metric = keras.metrics.MeanSquaredError(name="mse")


    def train_step(self, data):
        """
        Augmentation & Training Defined Here 
        """
        # Unpack data 
        obsv_mat = data[0]
        label_vec = data[1]
        # mos_label = tf.reshape(data['label'], [-1])

        with tf.GradientTape() as tape:
            # Forward Pass
            predict_vec = self(obsv_mat, training=True)
            
            # Compute Loss
            loss_value = self.loss(label_vec, predict_vec)

        # Compute gradients
        gradients = tape.gradient(
            loss_value,
            self.encoder.trainable_weights + self.predictor.trainable_weights 
        )

        # Update Weights
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.predictor.trainable_weights
            )
        )

        # Update Metrics
        self.metric.update_state(label_vec, predict_vec)
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        # Unpack data 
        obsv_mat = data[0]
        label_vec = data[1]
        # mos_label = tf.reshape(data['label'], [-1])

        # Forward Pass
        predict_vec = self(obsv_mat)

        self.metric.update_state(label_vec, predict_vec)
        return {m.name: m.result() for m in self.metrics}


    def call(self, obsv_mat, training=False):        
        ref_wav = obsv_mat[:, :, :, 0]
        deg_wav = obsv_mat[:, :, :, 1]            
        
        feature_mat = self.encoder(ref_wav, deg_wav)
        predict_vec = self.predictor(feature_mat, training=training)
        return predict_vec


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.metric,
        ]

