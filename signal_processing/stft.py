import numpy as np

def stft(y, fs, step=1, window=None, nfft=None):
    '''
    Coding..
    '''
    if window is None:
        window = np.ones(len(y))
    nw = len(window)
    if nfft is None:
        nfft = nw

    p = 0
    mat = np.empty((0, nfft//2 + 1), np.complex64)


    while p+nw<=len(y):
        
        shifted_window = np.concatenate((np.zeros(p, np.float), window, np.zeros(len(y) - (p+nw))))
        
        Y = np.fft.rfft((y * shifted_window)[p:], n=nfft)

        mat = np.append(mat, [Y], axis = 0)
        
        p += step

    #
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    time = np.arange(nw//2, len(y) - nw//2 + 1, step)/float(fs)

    #
    mat = mat.T
    mat *= np.sqrt(1.0 / window.sum()**2)
    
    #
    return (freqs, time, mat)