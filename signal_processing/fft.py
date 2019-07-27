import numpy as np

def slow_fft(signal, n=None):
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
    if n is None:
        n = signal.shape[0]

    # Adjust
    if n > signal.shape[0]:
        signal = np.append(signal, np.zeros(n - signal.shape[0]))
    signal = signal[:n]

    # Freq segments
    time = np.arange(n)
    output = np.zeros(n//2 + 1, np.complex)

    for f in range(n//2 + 1):
        # pi*-2j -> complete rotation
        # f -> frequency 1=one rotation
        # time/n -> transform time to 1 second range
        output[f] = sum(signal * np.exp(-2j * np.pi * f * time / n))
    #
    return output


def reduced_slow_fft(signal):
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

    n = signal.shape[0]

    a = np.exp(-2j * np.pi * np.arange(n//2 + 1).reshape((n//2+1, 1)) * np.arange(n) / n)

    return abs(np.dot(a, signal)) / n

def radix_2_fft(x):
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
    even = radix_2_fft(x[0::2])
    odd =  radix_2_fft(x[1::2])
    T = [np.exp(-2j*np.pi*k/N)*odd[k] for k in range(N//2)]
    R = [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]
    return np.asarray(R, np.float)