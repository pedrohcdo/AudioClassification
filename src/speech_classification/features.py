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
from ..signal_processing.audio_synthesizer import AudioSynthesizer

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(self.rate / 10)

class Features:
    def __init__(self, config):
        self.config = config
    
    @classmethod
    def extract_from(cls, config, dataset):
        feat = Features(config)
        x = []
        y = []
        _min, _max = float('inf'), -float('inf')
        for d in dataset:
            signal = d.data.compacted(20, 10, normalized=True, 
                                scale=AudioSynthesizer.COMPACT_SCALE_DENSITY).synthetized_signal()
            fs = d.data.fs
            label = d.label

            feat = mfcc(signal, fs, numcep=config.nfeat, 
                          nfilt=config.nfilt, nfft=config.nfft).T
                        
            _min = min(np.amin(feat), _min)
            _max = max(np.amax(feat), _max)

            x.append(feat if config.mode == 'conv' else feat.T)
            y.append(np.where(dataset.classes == label))

config = Config(mode='conv')


