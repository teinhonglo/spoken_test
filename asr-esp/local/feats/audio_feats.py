import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, fftconvolve
from scipy.interpolate import interp1d
import soundfile

import librosa
import librosa.display
import matplotlib.pyplot as plt

# 000360264 /share/corpus/speechocean762/WAVE/SPEAKER0036/000360264.WAV
wav_path = "/share/corpus/speechocean762/WAVE/SPEAKER0036/000360264.WAV"
speech, rate = soundfile.read(wav_path)

f0, voiced_flag, voiced_probs = librosa.pyin(speech,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))


S, phase = librosa.magphase(librosa.stft(speech))
rms = librosa.feature.rms(S=S)
# (143,)
print(f0.shape)
# (1, 143)
print(rms.shape)


# plot f0
times = librosa.times_like(f0)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')
plt.savefig("f0.png")
plt.figure().clear()
plt.close()
plt.cla()
plt.clf()

# plot energy
fig, ax = plt.subplots(nrows=2, sharex=True)
times = librosa.times_like(rms)
ax[0].semilogy(times, rms[0], label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='log Power spectrogram')
plt.savefig("energy.png")
