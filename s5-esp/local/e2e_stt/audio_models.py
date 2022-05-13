from scipy.io import wavfile
from scipy.signal import correlate, fftconvolve
from scipy.interpolate import interp1d

import librosa
import librosa.display

import os
import numpy as np
import json
import soundfile
from tqdm import tqdm
'''
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
'''
def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

def get_stats(numeric_list, prefix=""):
    # number, mean, standard deviation (std), median, mean absolute deviation
    stats_np = np.array(numeric_list)
    number = len(stats_np)
    
    if number == 0:
        summ = 0.
        mean = 0.
        std = 0.
        median = 0.
        mad = 0.
        maximum = 0.
        minimum = 0.
    else:
        summ = np.sum(stats_np)
        mean = np.mean(stats_np)
        std = np.std(stats_np)
        median = np.median(stats_np)
        mad = np.sum(np.absolute(stats_np - mean)) / number
        maximum = np.max(stats_np)
        minimum = np.min(stats_np)
    
    stats_dict = {  prefix + "number": number, 
                    prefix + "mean": mean, 
                    prefix + "std": std, 
                    prefix + "median": median, 
                    prefix + "mad": mad, 
                    prefix + "summ": summ,
                    prefix + "max": maximum,
                    prefix + "min": minimum
                 }
    return stats_dict
    
    
class AudioModel(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    
    def get_f0(self, speech):
        f0_list, voiced_flag, voiced_probs = librosa.pyin(speech, sr=self.sample_rate,
                                             frame_length=800,
                                             win_length=400, hop_length=160, center=False, 
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
        f0_list = np.nan_to_num(f0_list)
        f0_stats = get_stats(f0_list, prefix="f0_")
        f0_stats["f0_list"] = f0_list.tolist()
        f0_stats["f0_voiced_probs"] = voiced_probs.tolist()
        # removed unvoiced frames
        f0_nz_list = f0_list[np.nonzero(f0_list)]
        f0_nz_stats = get_stats(f0_nz_list, prefix="f0_nz_")
        f0_nz_stats["f0_nz_list"] = f0_nz_list.tolist()
        f0_stats = merge_dict(f0_stats, f0_nz_stats)
        
        return [f0_list, f0_stats]
    
    def get_energy(self, speech):
        rms = librosa.feature.rms(y=speech, frame_length=800, hop_length=160, center=False)
        rms_list = rms.reshape(rms.shape[1],)
        rms_stats = get_stats(rms_list, prefix="energy_")
        rms_stats["energy_rms_list"] = rms_list.tolist()
        
        return [rms_list, rms_stats]
    
if __name__ == "__main__":
    import soundfile
    # 414
    wav_path = "data/spoken_test_2022_jan28/wavs/0988973896/0988973896-3-1-2022_1_12.wav"
    speech, rate = soundfile.read(wav_path)
    assert rate == 16000
    
    audio_model = AudioModel(rate)
    # f0
    f0_list, f0_info = audio_model.get_f0(speech) 
    # energy 
    rms_list, rms_info = audio_model.get_energy(speech)
