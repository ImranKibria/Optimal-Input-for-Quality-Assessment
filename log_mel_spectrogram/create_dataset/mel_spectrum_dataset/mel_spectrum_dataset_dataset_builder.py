"""mel_spectrum_dataset dataset."""
import os
import scipy
import librosa
import dataclasses
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from voice_activity import find_vad

PATH_DS_BASE = '/fs/ess/PAS2301/Data/Speech/NISQA_Corpus'

@dataclasses.dataclass
class BuilderConfigEEG(tfds.core.BuilderConfig):
  sampling_rate: int = 16000
  n_fft: int = 512 
  n_mels: int = 57
  hop_len: int = 160
  win_len: int = 512
  normalize: bool = False

  frames = 25
  features = 58
  channels = 2


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mel_spectrum_dataset dataset."""

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
    # TODO(spectrogram_feature_dataset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
      features=tfds.features.FeaturesDict({
        'filename_deg':   tfds.features.Tensor(dtype=tf.string, shape=()),
        'feature_vector': tfds.features.Tensor(dtype=np.float32, shape=(self._builder_config.frames, self._builder_config.features, self._builder_config.channels)), 
        'label':          tfds.features.Tensor(dtype=np.float32,shape=(1,)), # MOS
      }),
    )


  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return {
      'NISQA_TRAIN_SIM': self._generate_examples('NISQA_TRAIN_SIM'),
      'NISQA_TRAIN_LIVE': self._generate_examples('NISQA_TRAIN_LIVE'),
      'NISQA_VAL_SIM': self._generate_examples('NISQA_VAL_SIM'),
      'NISQA_VAL_LIVE': self._generate_examples('NISQA_VAL_LIVE')
    }


  def get_valid_frames(self, ref_wav, deg_wav):
    fr_size = self._builder_config.win_len
    hop_len = self._builder_config.hop_len
    samp_rate = self._builder_config.sampling_rate
    
    # starting location of each frame
    fr_st_loc = range(0, len(deg_wav)-fr_size+1, hop_len) 
    tot_frames = len(fr_st_loc)
    val_frames = np.zeros(tot_frames)
    
    for fr_idx in range(0, tot_frames):
        curr_fr = deg_wav[fr_st_loc[fr_idx] : fr_st_loc[fr_idx] + fr_size]
        val_frames[fr_idx] = find_vad(curr_fr, samp_rate) # voice activity

    return val_frames


  @staticmethod
  def get_spectrum(wav, n_fft, hop_len, win_len, n_mels):
    wav_length = wav.shape[0]
    
    # fixing wav to a fixed size if needed
    wav_pad = wav # librosa.util.fix_length(data=wav, size=sr * fixed_len)  # signal_length + n_fft // 2)

    # mel-spectrogram computation 
    S = librosa.feature.melspectrogram(y=wav_pad, sr=16000, n_fft=n_fft, 
      hop_length=hop_len, win_length=win_len, window='hann', n_mels=n_mels)    
    
    return S


  def get_obsv_mat(self, ref_spect, deg_spect, valid_frames):
    frames = self._builder_config.frames
    features = self._builder_config.features
    channels = self._builder_config.channels
    
    paired_audio = np.stack([ref_spect, deg_spect], axis=2)

    tot_fr = np.shape(paired_audio)[0]
    val_fr_args = np.argwhere(valid_frames == 1).flatten()  # valid frame indices exceed voice activity threshold
    val_fr_args = val_fr_args[(val_fr_args >= 12) & (val_fr_args < tot_fr-12)] # first & last 12 frames not considered

    tot_obsv = np.size(val_fr_args)
    obsv_mat = np.zeros((tot_obsv, frames, features, channels))

    for m in range(tot_obsv):
        obsv_mat[m, :, :, :] = paired_audio[val_fr_args[m]-12 : val_fr_args[m]+12+1, :, :]

    return tot_obsv, obsv_mat


  def _generate_examples(self, split):
    """Yields examples."""
    # TODO(spectrogram_feature_dataset): Yields (key, example) tuples from the dataset

    n_fft = self._builder_config.n_fft
    n_mels = self._builder_config.n_mels
    hop_len = self._builder_config.hop_len
    win_len = self._builder_config.win_len
    normalize = self._builder_config.normalize
    sampling_rate = self._builder_config.sampling_rate

    corpus_path = PATH_DS_BASE + '/'
    df_train_sim = pd.read_csv(corpus_path + 'NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv')
    df_train_live = pd.read_csv(corpus_path + 'NISQA_TRAIN_LIVE/NISQA_TRAIN_LIVE_file.csv')

    df_train = pd.concat([df_train_sim, df_train_live], join='inner')

    df_valid_sim = pd.read_csv(corpus_path + 'NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv')
    df_valid_live = pd.read_csv(corpus_path + 'NISQA_VAL_LIVE/NISQA_VAL_LIVE_file.csv')

    df_valid = pd.concat([df_valid_sim, df_valid_live], join='inner')

    if split=='NISQA_TRAIN_SIM':
      df = df_train_sim
    elif split=='NISQA_TRAIN_LIVE':
      df = df_train_live
    elif split=='NISQA_VAL_SIM':
      df = df_valid_sim
    elif split=='NISQA_VAL_LIVE':
      df=df_valid_live

    for i, line in enumerate(df.itertuples()):
      path_ref = os.path.join(PATH_DS_BASE, line.filepath_ref)
      path_deg = os.path.join(PATH_DS_BASE, line.filepath_deg)

      ref_wav, _ = librosa.load(path_ref, sr=sampling_rate)
      deg_wav, _ = librosa.load(path_deg, sr=sampling_rate)
      
      valid_frames = self.get_valid_frames(ref_wav, deg_wav)

      ref_mel_spect = self.get_spectrum(ref_wav, n_fft=n_fft, hop_len=hop_len, win_len=win_len, n_mels=n_mels)
      deg_mel_spect = self.get_spectrum(deg_wav, n_fft=n_fft, hop_len=hop_len, win_len=win_len, n_mels=n_mels)

      tot_obsv, obsv_mat = self.get_obsv_mat(ref_mel_spect, deg_mel_spect, valid_frames)
      mos = np.asarray(line.mos)

      for n in range(tot_obsv):
        yield self.id, {
          'filename_deg':   line.filename_deg,
          'feature_vector': obsv_mat[n].astype(np.float32),
          'label':          mos.reshape((1,)).astype(np.float32),
        }
        self.id += 1

