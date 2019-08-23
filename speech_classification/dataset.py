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

class Dataset:
    
    DOWNSAMPLE_SR = 16000
    DOWNSAMPLE_SUB_FOLDER = "/ds"
    
    def __init__(self, folder, files):
        self.folder = folder
        self.files = np.array(files)
        self.downsampled = False
            
    def downsample(self):
        if self.downsampled:
            return
        self.downsampled_folder = self.folder + Dataset.DOWNSAMPLE_SUB_FOLDER
        try:
            os.mkdir(self.downsampled_folder)
        except OSError:
            self.downsampled = True
            print("--\nDownsampled! \nobs(For recreate downsampled files delete the '" + self.downsampled_folder + "' folder.)\n--")
            return
        for f in tqdm(self.files):
            signal, rate = librosa.load(self.folder + "/" + f, sr=16000)
            wavfile.write(filename=self.downsampled_folder + "/" + f, rate=rate, data=signal)
        self.downsampled = True
    
    def get_folder(self):
        if self.downsampled:
            return self.downsampled_folder
        return self.folder
    
    def load_file(self, filename):
        if self.downsampled:
            return librosa.load(self.get_folder() + "/" + filename, sr=Dataset.DOWNSAMPLE_SR)
        return librosa.load(self.get_folder() + "/" + filename)
        
    
class DFDataset(Dataset):
    
    FILENAME_COLUMN = 'fname'
    LABEL_COLUMN = 'label'

    Subset = collections.namedtuple("Subset", ["labels", "waves"], verbose=False, rename=False) 

    def __init__(self, df, folder, downsample=False, pruning_prop=0.0):
        # Check consistency
        assert DFDataset.FILENAME_COLUMN in df.columns and DFDataset.LABEL_COLUMN in df.columns
        #
        for _ in range(0, int(pruning_prop * len(df[DFDataset.FILENAME_COLUMN]))):
            df.drop(np.random.choice(df.index), inplace=True)

        #
        super().__init__(folder, df.fname)
        #
        if downsample:
            self.downsample()
        #
        self.df = df
        self.config()

    def config(self):
        print(self.df)
        self.df.set_index('fname', inplace=True)
        for filename in tqdm(self.df.index):
            signal, rate = self.load_file(filename)
            self.df.at[filename, 'length'] = signal.shape[0]/rate
        self.df.reset_index(inplace=True)

    def get_random(self, count, length_prob=1.0):
        assert length_prob<=1.0 + sys.float_info.epsilon and length_prob>=-sys.float_info.epsilon
        # Create prob distribution
        labels = []
        audios = []
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
            wave_piece = int(len(wave) * length_prob)
            rand_range = wave.shape[0] - wave_piece
            if rand_range > 0:
                rand_index = np.random.randint(0, wave.shape[0] - wave_piece)
                wave = wave[rand_index:rand_index+wave_piece]
            #
            labels.append(label)
            audios.append((wave, rate))
        return DFDataset.Subset(labels=labels, waves=audios)


        