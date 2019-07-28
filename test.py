import numpy as np
import simpleaudio as sa
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
from scipy import signal

#
from signal_processing.helper import AudioGenerator, play_audio
from signal_processing.fft import slow_fft, reduced_slow_fft, radix_2_fft
from signal_processing.stft import stft

# Generate audio with frequencies
freqs = np.array([10, 30, 70, 90])
time = 3
fs = 200

sound_gen = AudioGenerator(time, fs)
sound_gen.generate_signal(freqs)
sound_gen.generate_progressive_signal(20, 5, energy=0.2)
audio = sound_gen.generate()

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
    axes[1].plot(freqSeg, Y)
    
    # Plot slow fft
    Y = abs(slow_fft(signal, n=n * res))/n
    axes[2].set_title("Fourier Transform")
    axes[2].plot(freqSeg, Y)
    
    #
    plt.show()

# Compare fft methods
#compare_fft_methods(audio, fs)
#exit()



# Compare Short Time Fourier Transform 
nwin = 50
step = 10

f, t, Zxx = signal.stft(audio, fs=fs, noverlap=nwin-step, window=np.hanning(nwin), 
                        nperseg=nwin, boundary=None, padded=False, nfft=2000)

f2, t2, Zxx2 = stft(audio, fs, step=step, window=np.hanning(nwin), nfft=2000)

print(len(t))
print(len(t2))


plt.figure(1)
plt.pcolormesh(t, f, np.abs(Zxx))

plt.figure(2)
plt.pcolormesh(t2, f2, np.abs(Zxx2))
plt.show()
