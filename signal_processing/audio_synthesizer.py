import numpy as np
import simpleaudio as sa
from .utils import frame_signal

class AudioSynthesizer:
    
    def __init__(self, time, fs=44100, clamp_energy=False, normalize=True):
        self.time = time
        self.fs = fs
        self.samples = self.time * self.fs
        self.clamp_energy = clamp_energy
        self.normalize = normalize
        self.components = np.empty((0, self.time * self.fs), np.float)
        
    @classmethod
    def from_signal(cls, signal, fs):
        signal = np.array(signal).ravel()
        generator = cls(signal.shape[0] // fs, fs=fs)
        generator.components = np.append(generator.components, [signal[:generator.samples]], axis = 0)
        return generator
    
    def compact(self, precision, threshold, step=1):
        # Synthesise
        signals = self.synthesise()
        # Convolve and remove components below the threshold 
        framed = frame_signal(signals, np.ones((precision,)), step)
        convolved = np.dot(framed, np.ones((framed.shape[1],)))
        new_signal = convolved[abs(convolved)>threshold]
        # Reset attrs
        self.samples = new_signal.shape[0]        
        self.time = self.samples / self.fs
        # Set compacted signal
        self.components = np.array([new_signal], dtype=np.float)

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

    def synthesise(self):
        # Mix signals
        signals = np.zeros(self.samples, np.float)
        for component in self.components:
            signals += component
        
        # Clamp
        if self.clamp_energy:
            signals = np.clip(signals, a_min=-1, a_max=1)

        # Convert float signal to short signal
        scale = 1
        if self.normalize:
            scale = 1 / np.max(np.abs(signals))
        
        return signals * scale
    
    def generate(self):
        audio = self.synthesise() * (2**15 - 1)
        audio = audio.astype(np.int16)
        #
        return audio

