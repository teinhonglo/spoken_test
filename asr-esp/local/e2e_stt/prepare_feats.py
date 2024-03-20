import os
import json
import soundfile
from tqdm import tqdm
from espnet_models import SpeechModel
from audio_models import AudioModel
from vad_model import VadModel
from nlp_models import NlpModel
import numpy as np
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
max_segment_length = args.max_segment_length

output_dir = os.path.join(data_dir, model_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tag = args.model_tag
wavscp_dict = {}
text_dict = {}
utt_list = []
# stt and ctm
all_info = {}

speech_model = SpeechModel(tag)
audio_model = AudioModel(sample_rate)
vad_model = VadModel(mode=vad_mode, sample_rate=sample_rate, max_segment_length=max_segment_length)
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
    audio, rate = vad_model.read_wave(wav_path)
    speech = np.frombuffer(audio, dtype='int16').astype(np.float32) / 32768.0
    assert rate == sample_rate
    
    total_duration = speech.shape[0] / rate
    # audio feature
    _, f0_info = audio_model.get_f0(speech)
    _, energy_info = audio_model.get_energy(speech)
    # fluency feature and confidence feature
    speechs = vad_model.get_speech_segments(audio, rate)
    text = []
    for speech_seg in speechs:
        text_seg = speech_model.recog(speech_seg)
        text.append(text_seg)
    
    text = " ".join(" ".join(text).split())
    # alignment (stt)
    word_ctm_info = speech_model.get_ctm(speech, text)
    phn_ctm_info, phone_text = speech_model.get_phone_ctm(word_ctm_info)
    
    sil_feats_info, response_duration = speech_model.sil_feats(word_ctm_info, total_duration)
    word_feats_info, response_duration = speech_model.word_feats(word_ctm_info, total_duration)
    phone_feats_info, response_duration = speech_model.phone_feats(phn_ctm_info, total_duration)
    vp_feats_info = nlp_model.vocab_profile_feats(text)
    
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

