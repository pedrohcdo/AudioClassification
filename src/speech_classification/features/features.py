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
        self.training = []
        self.testing = []
        self.normalize_range = None
    
    @staticmethod
    def feature_by(config, signal, fs):
        if isinstance(config, STFT):
            nwin = config.window_len
            return ssignal.stft(signal, fs=fs, noverlap=nwin-config.step, window=config.window(nwin), 
                        nperseg=nwin, nfft=config.nfft)[2]
        elif isinstance(config, MFCC):
            return mfcc(signal, fs, numcep=config.nfeat, 
                          nfilt=config.nfilt, nfft=config.nfft, winlen=config.window_len, 
                          winstep=config.window_step, winfunc=config.window).T
        elif isinstance(config, FBANK):
            return logfbank(signal, fs, nfilt=config.nfilt, nfft=config.nfft).T

    @classmethod
    def extract_from(cls, config, dataset, max_len, normalize_range=None, testing_split=0.2):
        assert testing_split >= 0 and testing_split <= 1
        feat = Features(config)
        sets = {}
  

        _min, _max = float('inf'), -float('inf')
        for d in dataset:
            signal = d.data.synthetized_signal()
            fs = d.data.fs
            label = d.label
            
            if label not in sets:
                sets[label] = []


            feat_by = Features.feature_by(config, signal, fs)
            if (max_len > feat_by.shape[1]):
                pad_width = max_len - feat_by.shape[1]
                feat_by = np.pad(feat_by, pad_width=((0, 0), (0, pad_width)), mode='constant')
            feat_by = feat_by[:, :max_len]

            _min = min(np.amin(feat_by), _min)
            _max = max(np.amax(feat_by), _max)

            x = feat_by if config.mode == Config.MODE_CONV else feat_by.T
            y = dataset.classes.index(label)

            sets[label].append((x, y))

        training = []
        testing = []

        for key, values in sets.items():
            l = len(values)
            prop = int(l * testing_split)
            training.extend(values[:l-prop])
            testing.extend(values[l-prop:])


        assert len(training) > 0


        tx, ty = zip(*training)
        tsx, tsy = zip(*testing)

        TX, TY = np.array(tx), np.array(ty)
        TSX, TSY = np.array(tsx), np.array(tsy)

        #
        if normalize_range is not None:
            _min, _max = normalize_range
        # Normalize
        #X = (X - _min) / (_max - _min)
       
        if config.mode == Config.MODE_CONV:
            TX = TX.reshape(TX.shape + (1,))
            TSX = TSX.reshape(TSX.shape + (1,))
        elif config.mode == Config.MODE_DEEP:
            pass
        TY = to_categorical(TY, num_classes=10)
        TSY = to_categorical(TSY, num_classes=10)
        
        feat.normalize_range = (_min, _max)
        feat.training = (TX, TY)
        feat.testing = (TSX, TSY)
        
        return feat


