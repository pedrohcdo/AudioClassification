import numpy as np
import simpleaudio as sa

class AudioGenerator:
    
    def __init__(self, time, fs=44100, clamp_energy=False):
        self.time = time
        self.fs = fs
        self.samples = self.time * self.fs
        self.clamp_energy = clamp_energy
        self.components = np.empty((0, self.time * self.fs), np.float)

    def generate_progressive_signal(self, freq, inc, energy=1):
        r"""
        Generates progressive signal with the desired frequencies

        Parameters
        ----------
        freq : float
            Desired frequency
        angle : float 
            Progressive angle
        time : float
            Signal time
        fs : int, optional
            Sampling frequency

        Returns
        -------
        s : ndarray
            Generated signal
        """
        t = np.linspace(0, self.time, self.samples, False)
        signal = np.sin((freq + t * inc) * t * 2 * np.pi) * energy

        # Add component
        self.components = np.append(self.components, [signal], axis = 0)

    def generate_signal(self, frequencies, interpolate=False, energy=1):
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
            it = self.time / len(frequencies)
            t = np.linspace(0, it, it * self.fs, False)
            signal = np.array([], np.int)

            # Interpolate waves
            for freq in frequencies:
                signal = np.concatenate((signal, np.sin(freq * t * 2 * np.pi * energy)))

        else:
            t = np.linspace(0, self.time, self.samples, False)
            signal = np.zeros(t.shape[0], dtype=np.float64)

            # Sum waves
            for freq in frequencies:
                signal += np.sin(freq * t * 2 * np.pi) * energy
        
        # Add componet
        self.components = np.append(self.components, [signal], axis = 0)

    def generate(self):
        # Mix signals
        signals = np.zeros(self.samples, np.float)
        for component in self.components:
            signals += component
        
        # Clamp
        if self.clamp_energy:
            signals = np.clip(signals, a_min=-1, a_max=1)

        # Convert float signal to short signal
        audio = signals * (2**15 - 1) / np.max(np.abs(signals))
        audio = audio.astype(np.int16)
        #
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