
import numpy as np
from math import ceil

def stride_shape(signal, stride_len, step=1):
    # Recreate shape of last element and add new dimension
    # eg: (2, 3) = (2, a, b)
    # 
    # Fot the last element new shape is:
    # a = 3 - window + 1 
    # b = window 
    new_dim = ceil(((1 + signal.shape[-1] - stride_len)) / step)
    shape = signal.shape[:-1]
    shape += (new_dim, stride_len)
    return shape

def stride_signal(signal, stride_len, step=1):
    new_shape = stride_shape(signal, stride_len, step)
    # signal.strides[-1] * step -> to move steps
    # And repeat last stride -> to move items of row
    new_strides = signal.strides[:-1] + (signal.strides[-1] * step, signal.strides[-1],)
    #
    return np.lib.stride_tricks.as_strided(signal, shape=new_shape, strides=new_strides)

def frame_signal(signal, window, step, padded=True):
    """
    Return framed signal
    """
    if(step <= 0):
        raise RuntimeError("The step must have a value greater than or equal to 0")
    #
    step = int(step)
    window = np.asarray(window).reshape((-1,))
    signal = np.asarray(signal).reshape((-1,))

    window_len = window.shape[0]
    signal_len = signal.shape[0]
    # Pad signal
    if padded:
        s_shape = stride_shape(signal, window_len, step)
        pad_last = s_shape[0] * step
        if pad_last < signal_len:
            padd_steps = ceil((signal_len - pad_last) / step)
            pad_zeros = (pad_last + padd_steps * step + window_len - 1) - signal_len
            signal = np.concatenate((signal, np.zeros((pad_zeros,))))
    #
    frames = stride_signal(signal, stride_len=window_len, step=step)
    #
    return frames * window



def test_stride_signal():
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    strided = stride_signal(signal, 2, 2)

    for i in range(0, strided.shape[0]):
        for j in range(0, strided.shape[1]):
            strided[i][j] = 10 - (strided[i][j] - 1)
    
    ##
    assert np.array_equal(signal, np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))