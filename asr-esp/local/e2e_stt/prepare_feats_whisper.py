import os
import json
from tqdm import tqdm
from audio_models import AudioModel
from vad_model import VadModel
import numpy as np
import sys
import wave
import whisper
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer

import string
import jiwer
import torch
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

parser.add_argument("--language",
                    default="none",
                    type=str) 

parser.add_argument("--condition_on_previous_text", action="store_true", help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")

parser.add_argument("--suppress_numeric_tokens", action="store_true")

parser.add_argument("--suppress_punc_tokens", action="store_true")

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name
model_tag = args.model_tag
sample_rate = args.sample_rate
language = args.language
condition_on_previous_text = args.condition_on_previous_text
vad_mode = args.vad_mode
max_segment_length = args.max_segment_length
language = args.language

#encourage model to transcribe words literally
tokenizer = get_tokenizer(multilingual=False)  # use multilingual=True if using multilingual model

numeric_tokens = [
    i 
    for i in range(tokenizer.eot)
    if all(c in list("0123456789") for c in tokenizer.decode([i]).removeprefix(" "))
]

punc_tokens = [
    i 
    for i in range(tokenizer.eot)
    if all(c in list("!?-,.")  for c in tokenizer.decode([i]).removeprefix(" "))
]
# discussions/589
#punc_tokens = [0, 11, 13, 30]


suppress_tokens = [-1]

if args.suppress_numeric_tokens:
    suppress_tokens += numeric_tokens
    print("suppress numeric_tokens")

if args.suppress_punc_tokens:
    suppress_tokens += punc_tokens
    print("suppress punc_tokens")


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

speech_model = whisper.load_model(model_tag)
audio_model = AudioModel(sample_rate)
normalizer = EnglishTextNormalizer()

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
    # audio, rate = vad_model.read_wave(wav_path)
    # speech = np.frombuffer(audio, dtype='int16').astype(np.float32) / 32768.0
    # assert rate == sample_rate
    # total_duration = speech.shape[0] / rate
    # audio feature
    #_, f0_info = audio_model.get_f0(speech)
    #_, energy_info = audio_model.get_energy(speech)
    # fluency feature and confidence feature
    # speechs = vad_model.get_speech_segments(audio, rate)
    text_org = []
    all_tokens = []
    prompt_reset_since = 0
    decode_options = {"suppress_tokens": suppress_tokens}
    
    if language == "none":
        decode_options["language"] = None
    else:
        decode_options["language"] = language
    
    result = speech_model.transcribe(audio=wav_path, condition_on_previous_text = args.condition_on_previous_text, **decode_options)
    #pp.pprint(result)
    
    text_org = result["text"]
    text = normalizer(text_org)
    
    all_info[uttid] = { "stt": text,
                        "stt(punc)": text_org,
                        "prompt": text_prompt, 
                        "wav_path": wav_path
                      }
    if i % 50 == 0:
        print("text", text)
        print("text_org", text_org)
    

# write STT Result to file
with open(output_dir + "/text", "w") as fn:
    for uttid in utt_list:
        fn.write(uttid + " " + all_info[uttid]["stt"] + "\n")

# write STT Result to file
with open(output_dir + "/text.org", "w") as fn:
    for uttid in utt_list:
        fn.write(uttid + " " + all_info[uttid]["stt(punc)"] + "\n") 
