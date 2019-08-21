import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc, logfbank
import librosa

from .dataset import LabeledDataset

def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, 1 / rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.tight_layout(pad=1, w_pad=1, h_pad=5)
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            #axes[x,y].get_xaxis().set_visible(False)
            #axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

df = pd.read_csv('../instruments.csv')


# Prepare Dataset
ds = LabeledDataset(df.label, '../wavfiles', df.fname)
ds.downsample()

# Add length to df
df.set_index('fname', inplace=True)

for f in df.index:
    rate, signal = wavfile.read('../wavfiles/'+f)
    df.at[f, 'length'] = signal.shape[0]/rate

# Get all classes and mean size
classes = list(np.unique(df.label))
class_dist = df.groupby(['label'])['length'].mean()





print(ds.labels[0] + ", " + ds.files[0])
#fig, ax = plt.subplots()
#ax.set_title('Class Distribution', y=1.08)
#ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%',
#       shadow=False, startangle=90)
#ax.axis('equal')
#plt.show()

df.reset_index(inplace=True)


signals = {}
fft = {}
fbanks = {}
fbanks2 = {}
mfccs = {}

for c in classes:
    wav_file = df[df.label == c].iloc[0,0]
    #print(df[df.label == c])

    signal, rate = librosa.load('../wavfiles/' + wav_file, sr=44100)
    
    signals[c] = signal
    fft[c] = calc_fft(signal, rate)
    
    fbanks[c] = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T

    mfccs[c] = mfcc(signal, rate, numcep=13, nfilt=26, nfft=1103).T
    
#plot_signals(signals)
#plt.show()

#plot_fft(fft)
#plt.show()

plot_fbank(fbanks)
plt.show()