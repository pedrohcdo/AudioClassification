
import numpy as np
from math import ceil


def stride_signal(signal, slen, step=1):
    # Recreate shape of last element and add new dimension
    # eg: (2, 3) = (2, a, b)
    # 
    # Fot the last element new shape is:
    # a = 3 - window + 1 
    # b = window 
    new_dim = ceil(((1 + signal.shape[-1] - slen)) / step)
    new_shape = signal.shape[:-1]
    new_shape += (new_dim, slen)
    # Repeat last stride, because the step for each row is s=8*row
    new_strides = signal.strides[:-1] + (signal.strides[-1] * step, signal.strides[-1],)
    #
    return np.lib.stride_tricks.as_strided(signal, shape=new_shape, strides=new_strides)

def test_stride_signal():
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    strided = stride_signal(signal, 2, 2)

    for i in range(0, strided.shape[0]):
        for j in range(0, strided.shape[1]):
            strided[i][j] = 10 - (strided[i][j] - 1)
    
    ##
    assert np.array_equal(signal, np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]))

