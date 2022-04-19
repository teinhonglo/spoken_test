import soundfile as sf
import librosa
import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

parser.add_argument("--sample_rate",
                    default="16000",
                    type=str)

args = parser.parse_args()

data_dir = args.data_dir
wavscp_fn = os.path.join(data_dir, "wav.scp")
sample_rate = int(args.sample_rate)

utt_list = []
wavscp_dict = {}

# Added header to wavefile
print("Repaire: Added header to wavefile")
with open(wavscp_fn, "r") as fn:
    for line in tqdm(fn.readlines()):
        info = line.split()
        file_id, wav_path = info
        
        sig, sr = sf.read(wav_path)
        sf.write(wav_path, sig, sr)
        
        wavscp_dict[file_id] = wav_path
        utt_list.append(file_id)

print("Resample:", sample_rate)
for uttid in tqdm(utt_list):
    wav_path = wavscp_dict[uttid]
    speech, rate = librosa.load(wav_path, sr=sample_rate) # resample to 16 kHZ
    sf.write(wav_path, speech, rate)
