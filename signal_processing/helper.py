import numpy as np
import simpleaudio as sa

def generate_signal(frequencies, time, fs=44100, interpolate=False):
    r"""
    Generates a signal with the desired frequencies

    Parameters
    ----------
    frequencies : tuple or array_like
        Frequencies that will be included in the signal
    time : float
        Signal time
    fs : int, optional
        Sampling frequency

    Returns
    -------
    s : ndarray
        Generated signal
    """

    
    if interpolate:
        it = time / len(frequencies)
        t = np.linspace(0, it, it * fs, False)
        note = np.array([], np.int)

        # Interpolate waves
        for freq in frequencies:
            note = np.concatenate((note, np.sin(freq * t * 2 * np.pi)))

    else:
        t = np.linspace(0, time, time * fs, False)
        note = np.zeros(t.shape[0], dtype=np.float64)

        # Sum waves
        for freq in frequencies:
            note += np.sin(freq * t * 2 * np.pi)

    # Convert float signal to short signal
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)

    return audio

def play_audio(signal, fs):
    r"""
    Play an signal with 'simpleaudio'

    Parameters
    ----------
    signal : tuple or array_like
        Time series of measurement values
    fs : int, optional
        Sampling frequency
    """
    play_obj = sa.play_buffer(signal, 1, 2, fs)
    play_obj.wait_done()