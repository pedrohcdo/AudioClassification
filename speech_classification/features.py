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

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(self.rate / 10)

class FeatureGen:
    def __init__():
        
        
        pass
    
    def build_rand_feat():
        x = []
        y = []
        _min, _max = float('inf'), -float('inf')
        for _ in tqdm(range(n_samples)):
            rand_class = np.ranom.choice(class_dist.index, p=class_dist_prob)
            file = np.random.choice(df[df.label==rand_class].index)
            rate, wave = wavfile.read('../wavfiles/' + file)
            label = df.at[file, 'label']
            rand_index = np.random.randint(0, wav.shape[0] - config.step)
            x_sample = mfcc(sample, rate, numcep=config.nfeat, 
                          nfilt=config.nfilt, nfft=config.nfft).T
            _min = min(np.amin(x_sample), _min)
            _max = max(np.amax(x_sample), _max)
            x.append(x_sample if config.mod == 'conv' else x_sample.T)
            y.append(classes.index(label))

config = Config(mode='conv')


