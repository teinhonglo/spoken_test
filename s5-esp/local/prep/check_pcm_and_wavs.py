def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    speech, rate = soundfile.read(path)
    
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
    pcm_speech = np.frombuffer(pcm_data, dtype='int16').astype(np.float32) / 32768.0
    print("rate", rate, sample_rate)
    print("shape", speech.shape, pcm_speech.shape)
    print(speech)
    print(pcm_speech)
    print("data", (speech==pcm_speech).all())
    input()
    return pcm_data, sample_rate
