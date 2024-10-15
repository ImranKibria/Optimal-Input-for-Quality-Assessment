"""audio_feature_dataset dataset."""

import os
import librosa
import dataclasses
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import voice_activity as voice_activity

import warnings
warnings.filterwarnings("ignore")

corpus_path = '/fs/ess/PAS2301/Data/Speech/NISQA_Corpus'

@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
  sampling_rate: int = 16000
  hop_len: int = 160
  win_len: int = 512 
  normalize: bool = False

  frames = 25
  features = 58
  channels = 2
  

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for audio_feature_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }
  BUILDER_CONFIGS = [
    BuilderConfigEEG(name='default', description='signal and stft', sampling_rate = 16000,),
  ]
  
  id = 0

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(audio_feature_dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
      features=tfds.features.FeaturesDict({
        # These are the features of your dataset like images, labels ...
        'filename_deg':   tfds.features.Tensor(dtype=tf.string, shape=()),
        'feature_vector': tfds.features.Tensor(dtype=np.float32, shape=(self._builder_config.frames, self._builder_config.features, self._builder_config.channels)), 
        'label':          tfds.features.Tensor(dtype=np.float32, shape=(1,)), # MOS
      }),
    )

  
  # find features for ref/deg files for all their frames 
  def extract_audio_features(self, ref_data, deg_data, samp_rate):
    file_size = np.size(deg_data)                                       # total samples in each file
    fr_size = self._builder_config.win_len                              # frame size in num of samples 
    hop_size = self._builder_config.hop_len                             # hop size in num of samples 
    ftrs_in_fr = self._builder_config.features                          # features in 1 frame

    fr_st_loc = range(0, file_size-fr_size+1, hop_size)                 # starting location of each frame
    tot_fr = np.size(fr_st_loc)                                         # total frames in file

    ftr_data = np.zeros((tot_fr, ftrs_in_fr, 2))                        # index 0 - frame, index 1 - features, index 2 - ref/deg file 
    
    # File-wise feature extraction 
    for file_idx in range(2):
      if file_idx == 0:                                               # reference file index 
        file_data = ref_data
      elif file_idx == 1:                                             # degraded file index
        file_data = deg_data

      pitches, magnitudes = librosa.core.piptrack(y=file_data, sr=samp_rate, n_fft=fr_size, hop_length=hop_size, win_length=fr_size)
      mel_freq_coeff = librosa.feature.mfcc(y=file_data, sr=samp_rate, n_mfcc=26, n_fft=fr_size, hop_length=hop_size, win_length=fr_size)

      # Frame-wise feature extraction
      for fr_idx in range(1, tot_fr):                                 # starts from 1 for delta features
        prev_fr = file_data[fr_st_loc[fr_idx-1] : fr_st_loc[fr_idx-1] + fr_size]
        curr_fr = file_data[fr_st_loc[fr_idx] : fr_st_loc[fr_idx] + fr_size]

        index = magnitudes[:, fr_idx].argmax()
        pitch = pitches[index, fr_idx]                              # pitch

        vad = voice_activity.find_vad(curr_fr, samp_rate)           # voice activity
    
        energy = np.sum(curr_fr * curr_fr)                          # frame energy

        mfcc = mel_freq_coeff[:, fr_idx]                            # 26 mel frequency coefficients
        
        index = magnitudes[:, fr_idx-1].argmax()
        del_pitch = pitch - pitches[index, fr_idx-1]                # delta value of pitch

        del_vad = vad - voice_activity.find_vad(prev_fr, samp_rate) # delta value of voice activity
        
        del_energy = energy - np.sum(prev_fr * prev_fr)             # delta value of energy
        
        del_mfcc = mfcc - mel_freq_coeff[:, fr_idx-1]               # delta value of mel-freq coeffs

        ftr_data[fr_idx, :, file_idx] = np.concatenate(([pitch], [vad], [energy], mfcc, [del_pitch], [del_vad], [del_energy], del_mfcc))

    ftr_data = np.delete(ftr_data, 0, 0)                                # the loop indexing started at 1
    return ftr_data


  def convert_to_context_vectors(self, featr_mat):
    fr_per_obsv = self._builder_config.frames                           # frames in 1 observation
    ftrs_in_fr = self._builder_config.features                          # features in 1 frame
    tot_fr = np.shape(featr_mat)[0]
    
    val_fr_args = np.argwhere(featr_mat[:, 1, 1] == 1)                  # valid frame indices exceed voice activity threshold
    val_fr_args = val_fr_args.flatten()
    val_fr_args = val_fr_args[(val_fr_args >= 12) & (val_fr_args < tot_fr-12)]   # first & last 12 frames are not considered for obsvs

    tot_obsv = np.size(val_fr_args)
    obsv_mat = np.zeros((tot_obsv, fr_per_obsv, ftrs_in_fr, 2))         # 0-observation idx, 1-frame idx, 2-feature idx, 3-file idx

    # find observations corresponding to frame, features, file
    for i in range(tot_obsv):
      fr_idx = val_fr_args[i]
      obsv_mat[i, :, :, :] = featr_mat[fr_idx-12 : fr_idx+12+1, :, :]

    return obsv_mat, tot_obsv


  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(audio_feature_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
      'NISQA_TRAIN_SIM': self._generate_examples('NISQA_TRAIN_SIM'),
      'NISQA_TRAIN_LIVE': self._generate_examples('NISQA_TRAIN_LIVE'),
      'NISQA_VAL_SIM': self._generate_examples('NISQA_VAL_SIM'),
      'NISQA_VAL_LIVE': self._generate_examples('NISQA_VAL_LIVE')
    }


  def _generate_examples(self, dir_name):
    """Yields examples."""
    # TODO(audio_feature_dataset): Yields (key, example) tuples from the dataset
    sampling_rate = self._builder_config.sampling_rate
    
    label_file = corpus_path + '/' + dir_name + '/' + dir_name + '_file.csv'
    df = pd.read_csv(label_file)
    
    for i, line in enumerate(df.itertuples()):
      path_ref = os.path.join(corpus_path, line.filepath_ref)
      path_deg = os.path.join(corpus_path, line.filepath_deg)
      
      deg_data, _ = librosa.load(path=path_deg, sr=sampling_rate)
      ref_data, _ = librosa.load(path=path_ref, sr=sampling_rate)

      feature_mat = self.extract_audio_features(ref_data, deg_data, sampling_rate) 
      context_vectors, vector_count = self.convert_to_context_vectors(feature_mat)
      mos_label = np.asarray(line.mos)
 
      for n in range(vector_count):
        yield self.id, {
          'filename_deg':   line.filename_deg,
          'feature_vector': context_vectors[n].astype(np.float32),
          'label':          mos_label.reshape((1,)).astype(np.float32),
        }
        self.id += 1

