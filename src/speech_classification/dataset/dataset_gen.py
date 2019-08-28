#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:31:29 2019

@author: pedro
"""
from tqdm import tqdm
import pandas as pd
import librosa
from scipy.io import wavfile
import numpy as np
import os
import sys
import collections

from .samples import Samples
from .dataset import Dataset
from ...signal_processing.audio_synthesizer import AudioSynthesizer
    
class DFDatasetGenerator(Samples):
    
    FILENAME_COLUMN = 'fname'
    LABEL_COLUMN = 'label'

    def __init__(self, df, folder, downsample=False, pruning_prop=0.0):
        # Check consistency
        assert DFDatasetGenerator.FILENAME_COLUMN in df.columns and DFDatasetGenerator.LABEL_COLUMN in df.columns
        super().__init__(folder, df.fname)
        if downsample:
            self.downsample()
        self.classes = np.unique(df.label)
        #
        for _ in range(0, int(pruning_prop * len(df[DFDatasetGenerator.FILENAME_COLUMN]))):
            df.drop(np.random.choice(df.index), inplace=True)
        #
        self.df = df
        self.config()

    def config(self):
        self.df.set_index('fname', inplace=True)
        for filename in tqdm(self.df.index):
            signal, rate = self.load_file(filename)
            self.df.at[filename, 'length'] = signal.shape[0]/rate
        self.df.reset_index(inplace=True)

    def get_random(self, count, length=None, equalize_size=True):
        assert length == None or length>=0
        # Create prob distribution
        dataset = Dataset(self.classes)
        signals = []
        _min_l = float('inf')
        for _ in range(count):
            classes_mlength = self.df.groupby(['label'])['length'].mean()
            pdist = classes_mlength / classes_mlength.sum()
            #
            rand_label = np.random.choice(classes_mlength.index, p=pdist)
            filename = np.random.choice(self.df[self.df.label==rand_label].fname)
            #
            label = self.df.loc[self.df.fname==filename].label.item()
            wave, rate = self.load_file(filename)

            #
            wave_piece = min(wave.shape[0], length if length else wave.shape[0])
            rand_range = wave.shape[0] - wave_piece
            if rand_range > 0:
                rand_index = np.random.randint(0, wave.shape[0] - wave_piece)
                wave = wave[rand_index:rand_index+wave_piece]
            #
            signals.append((wave, rate))
            _min_l = min(_min_l, wave.shape[0])
        #
        equalized_signals = signals
        if equalize_size:
            equalized_signals = []
            for (wave, rate) in signals:
                equalized_signals.append((wave[:_min_l], rate))
        # 
        for (wave, rate) in equalized_signals:
            audio_synth = AudioSynthesizer.from_signal(wave, fs=rate)
            dataset.add_data(label=label, data=audio_synth)
        return dataset
        