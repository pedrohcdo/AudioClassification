import numpy as np

def slow_fft(signal, nfft=None):
    r"""
    For the purpose of study, for a faster method to use cache or more optimized 
    methods such as radix-2 FFT.

    Parameters
    ----------
    signal : tuple or array_like
        Time series of measurement values

    Returns
    -------
    s : ndarray
        Frequencies strength
    """
    signal = np.asarray(signal, dtype=float)
    if nfft is None:
        nfft = signal.shape[0]

    # Adjust
    # according to nyquist theorem, to get frequencies bigger than  signal size
    # need to extend the fs (or in other words extend signal) 
    if nfft > signal.shape[0]:
        signal = np.append(signal, np.zeros(nfft - signal.shape[0]))
    signal = signal[:nfft]

    # Freq segments
    time = np.arange(nfft)
    output = np.zeros(nfft//2 + 1, np.complex)

    for f in range(nfft//2 + 1):
        # pi*-2j -> complete rotation
        # f -> frequency 1=one rotation
        # time/n -> transform time to 1 second range
        output[f] = sum(signal * np.exp(-2j * np.pi * f * time / nfft))
    #
    return output


def reduced_slow_fft(signal, nfft=None):
    r"""
    For the purpose of study, for a faster method to use cache or more optimized 
    methods such as radix-2 FFT.

    Parameters
    ----------
    signal : tuple or array_like
        Time series of measurement values

    Returns
    -------
    s : ndarray
        Frequencies strength
    """
    signal = np.asarray(signal, dtype=float)
    if nfft is None:
        nfft = signal.shape[0]

    # Adjust
    if nfft > signal.shape[0]:
        signal = np.append(signal, np.zeros(nfft - signal.shape[0]))
    signal = signal[:nfft]

    a = np.exp(-2j * np.pi * np.arange(nfft//2 + 1).reshape((nfft//2+1, 1)) * np.arange(nfft) / nfft)

    return abs(np.dot(a, signal)) / nfft

def radix2_fft(x):
    r"""
    This function computes the one-dimensional discrete Fourier
    Transform (DFT) using Cooley–Tukey FFT Method (radix-2 FFT).

    Parameters
    ----------
    signal : tuple or array_like
        Time series of measurement values

    Returns
    -------
    s : ndarray
        Frequencies strength
    """
    N = len(x)
    if N <= 1: return x
    even = radix2_fft(x[0::2])
    odd =  radix2_fft(x[1::2])
    T = [np.exp(-2j*np.pi*k/N)*odd[k] for k in range(N//2)]
    R = [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]
    return np.asarray(R, np.float)


def powfft(frames, nfft):
    return 1.0 / nfft * np.square(reduced_slow_fft(frames, nfft))