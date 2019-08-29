#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:14:17 2019

@author: pedro
"""
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
from ...signal_processing.audio_synthesizer import AudioSynthesizer
from scipy import signal as ssignal

from .config import Config, STFT, MFCC, FBANK

class Features:
    def __init__(self, config):
        self.config = config
    
    @staticmethod
    def feature_by(config, signal, fs):
        if isinstance(config, STFT):
            nwin = config.window_len
            return ssignal.stft(signal, fs=fs, noverlap=nwin-config.step, window=config.window(nwin), 
                        nperseg=nwin, nfft=config.nfft)[2]
        elif isinstance(config, MFCC):
            return mfcc(signal, fs, numcep=config.nfeat, 
                          nfilt=config.nfilt, nfft=config.nfft).T
        elif isinstance(config, FBANK):
            return logfbank(signal, fs, nfilt=config.nfilt, nfft=config.nfft).T

    @classmethod
    def extract_from(cls, config, dataset):
        feat = Features(config)
        x = []
        y = []
        _min, _max = float('inf'), -float('inf')
        for d in dataset:
            signal = d.data.synthetized_signal()
            fs = d.data.fs
            label = d.label

            feat = Features.feature_by(config, signal, fs)
            
            _min = min(np.amin(feat), _min)
            _max = max(np.amax(feat), _max)

            x.append(feat if config.mode == Config.MODE_CONV else feat.T)
            y.append(np.where(dataset.classes == label)[0])
        
        X, Y = np.array(x), np.array(y)
        X = (X - _min) / (_max - _min)
       
        if config.mode == Config.MODE_CONV:
            X = X.reshape(X.shape + (1,))
        elif config.mode == Config.MODE_DEEP:
            pass
        Y = to_categorical(Y, num_classes=10)
        return X, Y

config = Config(mode='conv')


