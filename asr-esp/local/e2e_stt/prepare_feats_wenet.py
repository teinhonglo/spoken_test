import os
import json
import soundfile
from tqdm import tqdm
from audio_models import AudioModel
from vad_model import VadModel
import numpy as np
import sys
import wave
import wenetruntime as wenet

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

parser.add_argument("--model_tag",
                    default="Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave",
                    type=str)

parser.add_argument("--model_name",
                    default="gigaspeech",
                    type=str)
                    
parser.add_argument("--sample_rate",
                    default=16000,
                    type=int)

parser.add_argument("--vad_mode",
                    default=1,
                    type=int)

parser.add_argument("--max_segment_length",
                    default=15,
                    type=int)                    

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name
model_tag = args.model_tag
sample_rate = args.sample_rate
vad_mode = args.vad_mode

output_dir = os.path.join(data_dir, model_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tag = args.model_tag
wavscp_dict = {}
text_dict = {}
utt_list = []
# stt and ctm
all_info = {}

decoder = wenet.Decoder(model_tag,
                        lang='en',
                        nbest=5,
                        enable_timestamp=True)

audio_model = AudioModel(sample_rate)
vad_model = VadModel(vad_mode, sample_rate)

with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

with open(data_dir + "/text", "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:])

for i, uttid in tqdm(enumerate(utt_list)):
    wav_path = wavscp_dict[uttid]
    text_prompt = text_dict[uttid]
    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    with wave.open(wav_path, 'rb') as fin:
        assert fin.getnchannels() == 1
        audio = fin.readframes(fin.getnframes())
   
    # We suppose the wav is 16k, 16bits, and decode every 0.5 seconds
    chunk_wav = audio
    recog = decoder.decode(chunk_wav, True) 
    
    recog = json.loads(recog)
    if recog["type"] == "final_result":
        text = recog["nbest"][0]["sentence"].upper()
    
    
    all_info[uttid] = { "stt": text, "prompt": text_prompt,
                        "wav_path": wav_path}
    #all_info[uttid] = { "stt": text, "prompt": text_prompt,
    #                    "wav_path": wav_path, "ctm": ctm_info, 
    #                    "feats": {  **f0_info, **energy_info, 
    #                                **sil_feats_info, **word_feats_info,
    #                                **phone_feats_info,
    #                                "total_duration": total_duration,
    #                                "response_duration": response_duration}}
print(output_dir)

# write STT Result to file
with open(output_dir + "/text", "w") as fn:
    for uttid in utt_list:
        fn.write(uttid + " " + all_info[uttid]["stt"] + "\n")
