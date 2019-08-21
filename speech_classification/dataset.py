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

class Dataset:
    
    DOWNSAMPLE_SR = 16000
    DOWNSAMPLE_SUB_FOLDER = "/ds";
    
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
            print("For recreate downsampled files delete the '" + self.downsampled_folder + "' folder.")
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
        
    
class LabeledDataset(Dataset):
    
    def __init__(self, labels, folder, files):
        super().__init__(folder, files)
        assert len(labels) == len(files)
        self._labels = {}
        for i in range(len(files)):
            self._labels[files[i]] = labels[i]
    
    def load_file(self, filename):
        assert filename in self._labels
        return self._labels[filename], super().load_file(filename)
    



        