import os
import json
import soundfile
from tqdm import tqdm
from whisperx_models import SpeechModel
from audio_models import AudioModel
from nlp_models import NlpModel
import numpy as np
import sys
import wave
import whisper
from whisper.normalizers import EnglishTextNormalizer
import string
import jiwer
import torch

import argparse

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

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

parser.add_argument("--device",
                    default="cuda",
                    type=str)

parser.add_argument("--language",
                    default="none",
                    type=str) 

parser.add_argument("--condition_on_previous_text", action="store_true", help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

# do nothing, consistent to local/e2e_stt/prepare_feats_whisper.py
parser.add_argument("--suppress_numeric_tokens", action="store_true")

parser.add_argument("--suppress_punc_tokens", action="store_true")

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name
model_tag = args.model_tag
sample_rate = args.sample_rate
language = args.language
condition_on_previous_text = args.condition_on_previous_text
device = args.device

print(model_tag, language)
if condition_on_previous_text:
    print("used condition_on_previous_text")
else:
    print("not used condition_on_previous_text")

output_dir = os.path.join(data_dir, model_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tag = args.model_tag
wavscp_dict = {}
text_dict = {}
utt_list = []
# stt and ctm
all_info = {}

speech_model = SpeechModel(tag=model_tag, device=device, language=language, condition_on_previous_text=condition_on_previous_text)
audio_model = AudioModel(sample_rate)

normalizer = EnglishTextNormalizer()
nlp_model = NlpModel()

with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

with open(data_dir + "/text", "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:])

import pprint
pp = pprint.PrettyPrinter(indent=4)
for i, uttid in tqdm(enumerate(utt_list)):
    wav_path = wavscp_dict[uttid]
    text_prompt = text_dict[uttid]
    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    speech, rate = soundfile.read(wav_path)
    assert rate == sample_rate
    total_duration = speech.shape[0] / rate
    # audio feature
    _, f0_info = audio_model.get_f0(speech)
    _, energy_info = audio_model.get_energy(speech)
    # fluency feature and confidence feature
    # alignment (stt)
    
    try:
        text_result, ctm_results = speech_model.recog(wav_path)
    except:
        print(f"No audio are detected: {uttid} {wav_path}")
        continue
    
    text, text_norm = text_result
    word_ctm_info, phn_ctm_info = ctm_results
    
    sil_feats_info, response_duration = speech_model.sil_feats(word_ctm_info, total_duration)
    word_feats_info, response_duration = speech_model.word_feats(word_ctm_info, total_duration)
    phone_feats_info, response_duration = speech_model.phone_feats(phn_ctm_info, total_duration)
    vp_feats_info = nlp_model.vocab_profile_feats(text_norm)
    
    all_info[uttid] = { "stt": text, "prompt": text_prompt,
                        "wav_path": wav_path, 
                        "word_ctm": word_ctm_info, "ctm": phn_ctm_info, 
                        "feats": {  **f0_info, **energy_info, 
                                    **sil_feats_info, **word_feats_info,
                                    **phone_feats_info, **vp_feats_info,
                                    "total_duration": total_duration,
                                    "response_duration": response_duration}}
    
    if i % 1000 == 0:
        print(all_info[uttid])


print(output_dir)
with open(output_dir + "/all.json", "w") as fn:
    json.dump(all_info, fn, indent=4, ensure_ascii=False, cls=NpEncoder)

# write STT Result to file
with open(output_dir + "/text", "w") as fn:
    for uttid in utt_list:
        if uttid in all_info:
            fn.write(uttid + " " + all_info[uttid]["stt"] + "\n")

