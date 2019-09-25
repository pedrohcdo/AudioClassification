import numpy as np

class Config:
    MODE_CONV = 'conv'
    MODE_DEEP = 'deep'

    def __init__(self, mode):
        assert mode in [Config.MODE_CONV, Config.MODE_DEEP]
        self.mode = mode

class STFT(Config):
    def __init__(self, mode, nfft=512, window=np.hamming, window_len=100, step=100):
        super().__init__(mode)
        self.nfft = nfft
        self.window = window
        self.window_len = window_len
        self.step = step

class FBANK(Config):
    def __init__(self, mode, nfilt=26, nfft=512):
        super().__init__(mode)
        self.nfilt = nfilt
        self.nfft = nfft

class MFCC(Config):
    def __init__(self, mode, nfilt=26, nfft=512, nfeat=14, window_len=0.025, window_step=0.005, window=lambda x:np.ones((x,))):
        super().__init__(mode)
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.window_len = window_len
        self.window_step = window_step
        self.window = window