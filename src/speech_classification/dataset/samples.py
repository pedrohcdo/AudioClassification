from tqdm import tqdm
import pandas as pd
import librosa
from scipy.io import wavfile
import numpy as np
import os
import sys
import collections

class Samples:
    
    DOWNSAMPLE_SR = 8000
    DOWNSAMPLE_SUB_FOLDER = "/ds"
    
    def __init__(self, folder, files):
        self.folder = folder
        self.files = np.array(files)
        self.downsampled = False
            
    def downsample(self):
        if self.downsampled:
            return
        self.downsampled_folder = self.folder + Samples.DOWNSAMPLE_SUB_FOLDER
        try:
            os.mkdir(self.downsampled_folder)
        except OSError:
            self.downsampled = True
            print("--\nDownsampled! \nobs(For recreate downsampled files delete the '" + self.downsampled_folder + "' folder.)\n--")
            return
        for f in tqdm(self.files):
            signal, rate = librosa.load(self.folder + "/" + f, sr=Samples.DOWNSAMPLE_SR)
            wavfile.write(filename=self.downsampled_folder + "/" + f, rate=rate, data=signal)
        self.downsampled = True
    
    def get_folder(self):
        if self.downsampled:
            return self.downsampled_folder
        return self.folder
    
    def load_file(self, filename):
        if self.downsampled:
            return librosa.load(self.get_folder() + "/" + filename, sr=Samples.DOWNSAMPLE_SR)
        return librosa.load(self.get_folder() + "/" + filename)
        