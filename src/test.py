'''
import numpy as np
import simpleaudio as sa
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
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
from .speech_classification.dataset import DFDatasetGenerator
from .signal_processing.helper import play_audio
from .signal_processing.utils import frame_signal
from .speech_classification.features import MFCC, STFT, FBANK, Features
import librosa
from scipy import signal
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, MaxPool2D, Dropout, Flatten

from .signal_processing.stft import stft
from sklearn.utils.class_weight import compute_class_weight


def get_conv_model(input_shape, classes_size):
    model = Sequential()
    #
    model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    # 
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.3))
    #
    model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape))
    #
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes_size, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['acc'])
    return model



data_gen = DFDatasetGenerator(pd.read_csv('./instruments.csv'), './wavfiles', downsample=True, pruning_prop=0.3)
dataset = data_gen.get_random(10, length=4000, equalize_size=True)

X1, Y1 = Features.extract_from(STFT(STFT.MODE_CONV), dataset)
#X2, Y2 = Features.extract_from(MFCC(MFCC.MODE_CONV), dataset)
#X3, Y3 = Features.extract_from(FBANK(FBANK.MODE_CONV), dataset)

y_flat = np.argmax(Y1, axis=1)
cw = compute_class_weight('balanced', np.unique(y_flat), y_flat)

model = get_conv_model((X1.shape[1], X1.shape[2], 1), len(dataset.classes))

model.fit(X1, Y1, epochs=1000, batch_size=32, shuffle=True, class_weight=cw)

print(model.predict(X1[0].reshape(1, X1.shape[1], X1.shape[2], 1), batch_size=32))
print(Y1[0])

exit()

for data in dataset:
    original = data.data
    compacted = data.data.compacted(20, 10, normalized=True, 
                                scale=AudioSynthesizer.COMPACT_SCALE_DENSITY)

    plt.plot(original.synthetized_signal())
    plt.plot(compacted.synthetized_signal())
    plt.show()

    play_audio(original)
    play_audio(compacted)





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
