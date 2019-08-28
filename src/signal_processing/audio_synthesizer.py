import numpy as np
import simpleaudio as sa
from .utils import frame_signal

class AudioSynthesizer:
    
    COMPACT_SCALE_NONE = 'none'
    COMPACT_SCALE_MAX = 'max'
    COMPACT_SCALE_DENSITY = 'density'

    def __init__(self, samples, fs=44100, clamp_energy=False, normalize=True):
        self.samples = samples
        self.time = samples / fs
        self.fs = fs
        self.clamp_energy = clamp_energy
        self.normalize = normalize
        self.components = np.empty((0, self.samples), np.float)
        
    @classmethod
    def from_signal(cls, signal, fs):
        signal = np.array(signal).ravel()
        generator = cls(len(signal), fs=fs)
        generator.components = np.append(generator.components, [signal], axis = 0)
        return generator
    
    def compacted(self, precision, threshold, normalized=False, padded=True, scale='none'):
        # Synthesise
        signals = self.synthetized_signal()
        # Convolve and remove components below the threshold 
        framed = frame_signal(signals, np.ones((precision,)), 1, padded=padded)
        #convolved = np.dot(abs(framed), np.ones((framed.shape[1],)).reshape(-1,1)).reshape((-1,))
        convolved = np.sum(abs(framed), axis=1)
        assert scale in [AudioSynthesizer.COMPACT_SCALE_NONE, 
                        AudioSynthesizer.COMPACT_SCALE_DENSITY, 
                        AudioSynthesizer.COMPACT_SCALE_MAX]
        if scale == AudioSynthesizer.COMPACT_SCALE_MAX:
            convolved = convolved / np.max(convolved)
        elif scale == AudioSynthesizer.COMPACT_SCALE_DENSITY:
            convolved = (convolved * len(signals)) / sum(abs(signals))
        new_signal = signals[:len(convolved)][convolved>threshold]
        #
        return AudioSynthesizer.from_signal(new_signal, fs=self.fs)

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

    def synthetized_signal(self):
        # Mix signals
        signals = np.zeros(self.samples, np.float)
        for component in self.components:
            signals += component
        
        # Clamp
        if self.clamp_energy:
            signals = np.clip(signals, a_min=-1, a_max=1)
        
        return signals
    
    def generate(self):
        # Convert float signal to short signal
        signals = self.synthetized_signal()

        scale = 1
        if self.normalize:
            scale = 1 / np.max(np.abs(signals))

        audio = signals * (2**15 - 1) * scale
        audio = audio.astype(np.int16)
        #
        return audio

