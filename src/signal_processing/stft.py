import numpy as np
from .utils import frame_signal

def stft(y, fs, step=1, window=None, nfft=None, psd=False):
    '''
    Coding..
    '''
    if window is None:
        window = np.ones(len(y))
    nw = len(window)
    if nfft is None:
        nfft = nw

    mat = np.empty((0, nfft//2 + 1), np.complex64)

    framed_signal = frame_signal(y, window, step, padded=False)
    
    mat = np.fft.rfft(framed_signal, n=nfft)

    #
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    time = np.arange(nw//2, len(y) - nw//2 + 1, step)/float(fs)

    #
    if psd:
        mat = mat * np.conjugate(mat)
    #
    mat = mat.T
    mat *= np.sqrt(1.0 / window.sum()**2)

    #mat = (mat**2) / fs
    
    #ref = 32768
    #s_mag = np.abs(mat) * 2 / window.sum()
    #mat = 2595 * np.log10(mat / 700 + 1)

    #
    return (freqs, time, mat)