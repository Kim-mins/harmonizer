import EBCTPS
import soundfile as sf
import numpy as np


def shift_pitch(data, Fs, ratio):
  zff = 0

  # scale = key ratio
  #scale = 1.2
  if len(data.shape) > 1:
    data = np.array([data[i][0] for i in range(len(data))])

  zff = EBCTPS.epoch(data, Fs)

  scaled_sp = EBCTPS.EPS(data, Fs, ratio, zff)

  return scaled_sp

  #sf.write('./'+location[:-4]+'_t.wav', scaled_sp, Fs)
