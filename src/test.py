'''
import numpy as np
import simpleaudio as sa
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from scipy import signal
'''
#
#from signal_processing.audio_synthesizer import AudioSynthesizer
#from signal_processing.helper import play_audio

'''

from signal_processing.fft import slow_fft, reduced_slow_fft, radix2_fft
from signal_processing.stft import stft

import librosa



from speech_classification.dataset import LabeledDataset
'''
#%%
import sys

# Fo run in jupyter extension
import src
__package__ = 'src'

#
import pandas as pd
from scipy.io import wavfile
import numpy as np

from .signal_processing.audio_synthesizer import AudioSynthesizer
from .speech_classification.dataset import DFDataset
from .signal_processing.helper import play_audio
from .signal_processing.utils import frame_signal
import librosa
from scipy import signal
import matplotlib.pyplot as plt

df_dataset = DFDataset(pd.read_csv('./instruments.csv'), './wavfiles', downsample=True, pruning_prop=0.3)
result = df_dataset.get_random(10)
for data in result:
    plt.plot(data.audio.synthetized_signal())
    plt.plot(data.audio.compacted(100, 0.3).synthetized_signal())
    plt.show()







'''
# 


# Add length to df





a = AudioSynthesizer(5,8000)
a.generate_progressive_signal(2000, 1000/5)
a.compact(1, -1, step=1)
play_audio(a)


a = AudioSynthesizer(5,8000)
a.generate_progressive_signal(2000, 1000/5)
a.compact(20, 0.3, step=1)
play_audio(a)

exit()

'''
'''


# Prepare Dataset
ds = LabeledDataset(df.label, './wavfiles', df.fname)
ds.downsample()


label, (signal, rate) = ds.load_file('5388d14d.wav')

ag = AudioSynthesizer.from_signal(signal, fs=rate)
ag.normalize = False

play_audio(ag)


print(label)
print(rate)
print(np.max(signal))
'''


'''
#

# Generate audio with frequencies
freqs = np.array([80])
time = 3
fs = 200

sound_gen = AudioGenerator(time, fs)
sound_gen.generate_signal(freqs)
#sound_gen.generate_progressive_signal(20, 5, energy=0.2)
audio = sound_gen.generate()

#audio, fs = librosa.load('wavfiles/0ed06544.wav')
#audio = audio[:fs]
#time = 1

# 
# play_audio(audio, fs)

#
def compare_fft_methods(signal, fs):
    fig, axes = plt.subplots(nrows=3, sharex=False,
                             sharey=True, figsize=(20,10))
    fig.suptitle('Graphs', size=16)

    n = len(signal)
    res = 100

    #
    freqSeg = np.fft.rfftfreq(n * res, 1 / fs)

    # Plot signal
    axes[0].set_title("Mixed signal")
    axes[0].plot(signal)

    # Plot fast fft
    Y = abs(np.fft.rfft(signal, n=n * res))/n
    axes[1].set_title("Fourier Transform")
    #axes[1].plot(freqSeg, Y)
    axes[1].imshow(freqSeg,
                    cmap='hot', interpolation='nearest')
    
    # Plot slow fft
    Y = abs(slow_fft(signal, n=n * res))/n
    axes[2].set_title("Fourier Transform")
    axes[2].imshow(freqSeg,
                    cmap='hot', interpolation='nearest')
    #axes[2].plot(freqSeg, Y)
    
    #
    plt.show()

# Compare fft methods
#compare_fft_methods(audio, fs)
#exit()

#audio = np.append(audio[0], audio[1:] - 1 * audio[:-1])

# Compare Short Time Fourier Transform 
nwin = int(fs * 0.5) # 25ms
step = int(fs * 0.1) # 10ms

f, t, Zxx = signal.stft(audio, fs=fs, noverlap=nwin-step, window=np.hamming(nwin), 
                        nperseg=nwin, boundary=None, padded=False, nfft=nwin)

f2, t2, Zxx2 = stft(audio, fs, step=step, window=np.hamming(nwin), nfft=100)

print(len(t))
print(len(t2))


plt.figure(1)
plt.pcolormesh(t, f, np.abs(Zxx), cmap='hot')

plt.figure(2)
plt.pcolormesh(t2, f2, np.abs(Zxx2), cmap='hot')
plt.show()
'''
