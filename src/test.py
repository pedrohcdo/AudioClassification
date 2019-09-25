

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
from keras.layers import Conv2D, Dense, Activation, MaxPool2D, Dropout, Flatten, UpSampling2D

from keras.callbacks import ModelCheckpoint

from .signal_processing.stft import stft
from sklearn.utils.class_weight import compute_class_weight
from keras import optimizers
from keras.models import load_model
import keras
from keras.constraints import max_norm


def get_conv_model(input_shape, classes_size):
    model = Sequential()
    #
    #model.add(UpSampling2D(size=(2,2), interpolation='bilinear', input_shape=input_shape))
    model.add(Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=input_shape,
            kernel_regularizer=keras.regularizers.l2(0.01), 
            bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(Conv2D(64, (2, 2), activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01), 
            bias_regularizer=keras.regularizers.l2(0.01)))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(rate=0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', 
            kernel_regularizer=keras.regularizers.l2(0.01), 
            bias_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(rate=0.25))
    model.add(Dense(classes_size, activation='softmax'))
    model.summary()

    sgd = optimizers.SGD(lr=0.004, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
    return model



def train_model(filepath):
    data_gen = DFDatasetGenerator(pd.read_csv('./instruments.csv'), './wavfiles', downsample=True, pruning_prop=0)
    #dataset = data_gen.get_random(20, length=2000, equalize_size=True)
    dataset = data_gen.get_random_on_classes(30, length=15000)

    print(dataset.classes)

    exit()

    #X1, Y1 = Features.extract_from(STFT(STFT.MODE_CONV), dataset)
    feat = Features.extract_from(MFCC(MFCC.MODE_CONV, window=lambda x: np.hamming(x)), dataset, max_len=80, testing_split=0.01)

    TX, CTY, TY = feat.training
    TSX, CTSY = feat.testing

    nr = feat.normalize_range



    #X3, Y3 = Features.extract_from(FBANK(FBANK.MODE_CONV), dataset)

    #y_flat = np.argmax(Y1, axis=1)
    #cw = compute_class_weight('balanced', np.unique(TY), TY)

    #model = get_conv_model((TX.shape[1], TX.shape[2], 1), len(dataset.classes))

    #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    #model.fit(TX, CTY, epochs=200000, batch_size=32, shuffle=False, class_weight=cw,
    #            callbacks=callbacks_list, validation_data=(TSX, CTSY))
    #print(nr)
    #exit()


def test_model(filepath):
    df = pd.DataFrame([['violin.mp3', 'violin'],
                       ['flute.wav', 'flute'],
                       ['aguitar.mp3', 'aguitar'],
                       ], columns=['fname', 'label'])

    data_gen = DFDatasetGenerator(df, './', downsample=True)
    dataset = data_gen.get_random_on_classes(length=4000)
    feat = Features.extract_from(MFCC(MFCC.MODE_CONV, window=lambda x: np.hamming(x)), dataset, max_len=80, testing_split=0.01, normalize_range=(-120.84730734274068, 109.68556357548314))

    TX, CTY, TY = feat.training
    TSX, CTSY = feat.testing

    model = load_model(filepath)
    model_labels = ['Acoustic_guitar', 'Bass_drum', 'Cello', 'Clarinet', 'Double_bass', 'Flute', 'Hi-hat', 'Saxophone', 'Snare_drum', 'Violin_or_fiddle']

    p = model.predict(TX, batch_size=32)
    mx = np.argmax(p, axis=1)

    for i in range(mx.shape[0]):
        print("Real: {} | Predict: {}".format( dataset.classes[TY[i]], model_labels[mx[i]] ))



test_model("first_model.h5")


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
