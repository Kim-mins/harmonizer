import numpy as np
import aubio
import librosa
import config
import soundfile as sf

from pitch_shift import shift_pitch


def round_note(freq):
  # round_note: get the nearest note (=rounded note) from dominant frequency (estimated frequency from pitch detection)
  if freq != 0.0:
    fs = list(config.freq2note.keys())
    fs_sub = list(map(lambda f: abs(freq - float(f)), fs))
    _, min_idx = min(((val, idx) for (idx, val) in enumerate(fs_sub)))
    # get the nearest note
    return config.freq2note[fs[min_idx]]
  else:
    # if freq = 0
    return 'N'


def get_note_ratio(note, target_notes):
  # get_note_ratio: get ratios between chord notes' and rounded note
  notes = list(config.note2freq.keys())
  f, s, t = target_notes
  i = notes.index(note)
  def ratio(x): return 2 ** (x / 12)
  f = ratio(notes.index(f) - i)
  s = ratio(notes.index(s) - i)
  t = ratio(notes.index(t) - i)
  return (f, s, t)


def get_target_keys(note, lab):
  # get_target_keys: get target notes from chord key & rounded note
  # assumes target frequency has the same octave with the rounded note
  # 옥타브가 바뀌어서인지 음이 계속 바뀌는 경우가 계속 생김
  if note == 'N' or lab == 'N':
    return (0.0, 0.0, 0.0)  # no chord
  octave = note[-1]
  # check minor/major and get corresponding notes
  f, s, t = config.minor_triad[lab] if 'min' in lab else config.major_triad[lab]
  return get_note_ratio(note, (f+octave, s+octave, t+octave))


def get_idx_and_note(data):
  # get_idx_and_note: result of tracking index and note of original song
  idx_and_notes = []
  note_prev = ''
  print("detecting pitch of original song...")
  for i, frame in enumerate(data):
    sample_idx = i * p.hop_size  # sample index
    note = round_note(p(frame)[0])  # dominant note (corrected pitch)
    print('\r%.2f%%' % (i/len(data)*100), end="")
    if note != note_prev:
      idx_and_notes.append((sample_idx, note))
      note_prev = note
  print("")
  return idx_and_notes


def get_label(name, sr):
  # get_label: get label from .lab file
  lab = []
  with open(name+".lab", "r") as f:
    while True:
      line = f.readline()[:-1]
      if line is "":
        break
      else:
        start, _, key = line.split()
        start = int(float(start)*sr)
        lab.append((start, key))
  return lab


def mix_data(idx_and_notes, label, sr):
  idx_and_notes = [(idx, note, "N") for (idx, note) in idx_and_notes]
  label = [(i, "N", lab) for (i, lab) in label]
  mixed_data = sorted(idx_and_notes + label)
  for i, (_, note, lab) in enumerate(mixed_data):
    if i >= 0:
      if note == "N":
        mixed_data[i] = (mixed_data[i][0], mixed_data[i-1]
                         [1], mixed_data[i][2])
      elif lab == "N":
        mixed_data[i] = (mixed_data[i][0], mixed_data[i]
                         [1], mixed_data[i-1][2])

  res = []
  MIN_INTERVAL = int(np.floor(0.03 * sr))
  for i, (idx, _, _) in enumerate(mixed_data):
    interval = 0
    if i != len(mixed_data)-1:
      interval = mixed_data[i+1][0]-idx
    if MIN_INTERVAL < interval:
      res.append(mixed_data[i])

  return res


def get_chord_data(y, sr, ratios):
  res = np.array([])
  end = len(ratios)-1
  print("chording...")
  for i, (idx, (f, s, t)) in enumerate(ratios):
    print('\r%.2f%%' % (i/len(ratios)*100), end="")
    data = None
    if i != end:
      data = y[idx:ratios[i+1][0]]
      # print(i, ratios[i+1])
    else:
      data = y[idx:y.shape[0]]
    if f == 0.0 and s == 0.0 and t == 0.0:
      res = np.concatenate((res, data*3))
    else:
      f = shift_pitch(data, sr, f)
      s = shift_pitch(data, sr, s)
      t = shift_pitch(data, sr, t)
      if len(f) < len(data):
        f = np.append(f, f[-1])
      if len(s) < len(data):
        s = np.append(s, s[-1])
      if len(t) < len(data):
        t = np.append(t, t[-1])
      res = np.concatenate((res, (f+s+t)))
  return res


if __name__ == '__main__':
  name = 'ㅈㅅㅁㅇㄴ'
  # load data
  # y, sr = sf.read(name+'.wav')
  y, sr = librosa.load(name+'.wav')
  label = get_label(name, sr)
  # create pitch object (for pitch detection)
  p = aubio.pitch("default", samplerate=sr)
  # pad end of input vector with zeros
  pad_length = p.hop_size - y.shape[0] % p.hop_size
  y_padded = np.pad(y, (0, pad_length), 'constant', constant_values=0)
  # to reshape it in blocks of hop_size
  y_padded = y_padded.reshape(-1, p.hop_size)
  # input array should be of type aubio.float_type (defaults to float32)
  y_padded = y_padded.astype(aubio.float_type)
  # get sample index and note from original song
  idx_and_notes = get_idx_and_note(data=y_padded)
  # get mixed data
  mixed_data = mix_data(idx_and_notes, label, sr)
  # get corresponding chord ratios from chord label
  ratios = [(idx, get_target_keys(note, lab))
            for (idx, note, lab) in mixed_data]
  # get chord data
  chord_data = get_chord_data(y, sr, ratios)

  # sf.write('./'+name+'_t.wav', chord_data, sr)
  librosa.output.write_wav(name+"_t.wav", chord_data, sr)

  # new_signal = shift_pitch(orig_signal, sr, f_ratio)

  # librosa.output.write_wav(
  #    name+"_{:01.2f}.wav".format(f_ratio), new_signal, sr)
