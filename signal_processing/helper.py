import simpleaudio as sa

def play_audio(audio_generator):
    r"""
    Play an signal with 'simpleaudio'

    Parameters
    ----------
    signal : tuple or array_like
        Time series of measurement values
    fs : int, optional
        Sampling frequency
    """
    play_obj = sa.play_buffer(audio_generator.generate(), 1, 2, audio_generator.fs)
    play_obj.wait_done()