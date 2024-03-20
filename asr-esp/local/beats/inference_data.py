import torch
import torchaudio
import os
from tqdm import tqdm
import numpy as np
import csv
from BEATs import BEATs, BEATsConfig

def speech_file_to_array_fn(path, target_sampling_rate=16000):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

# load the fine-tuned checkpoints
checkpoint = torch.load('pretrained_models/beats_iter3_plus/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt')
audio_csv_path = "pretrained_models/beats_iter3_plus/audioset_label.csv"

audioset_dict = {}
with open(audio_csv_path, "r") as fn:
    csvreader = csv.reader(fn)
    for row in csvreader:
        idx, tag, display_name = row
        audioset_dict[tag] = display_name

for i, tag in checkpoint["label_dict"].items():
    checkpoint["label_dict"][i] = audioset_dict[tag]

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()

anno_fn = "data/ccs2020-21/text.anno"
wavscp_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid2/wav.scp"
pred_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid2/text"
valid_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid2/wav.btvalid.scp"
valid_at_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid2/wav.bt.scp"
valid_atflt_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid2/wav.btflt.scp"
valid_speech_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid2/wav.btspeech.scp"
valid_nospeech_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid2/wav.btnospeech.scp"
speech_set = set(["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking"])

anno_dict = {}
wav_dict = {}
pred_dict = {}

valid_text_id = []
valid_at_text_id = []
valid_atflt_text_id = []
valid_speech_text_id = []
valid_nospeech_text_id = []
decode_options = {"language": "zh"}

print("annotation", anno_fn)
with open(anno_fn, "r") as fn:
    for line in tqdm(fn.readlines()):
        text_id, anno_res = line.split()
        anno_dict[text_id] = anno_res

print("wavscp_fn", wavscp_fn)
with open(wavscp_fn, "r") as fn:
    for line in tqdm(fn.readlines()):
        text_id, wav_path = line.split()
        wav_dict[text_id] = wav_path

print("pred_fn", pred_fn)
with open(pred_fn, "r") as fn:
    for line in tqdm(fn.readlines()):
        info = line.split()
        text_id = info[0]
        
        res_len = 0
        if len(info) > 1:
            res_len = len(info[1:])

        pred_dict[text_id] = res_len


if os.path.exists(valid_fn):
    print("valid fn is already existed", valid_fn)
    with open(valid_fn, "r") as fn:
        for line in fn.readlines():
            text_id, wav_path = line.split()
            valid_text_id.append(text_id)

    with open(valid_at_fn, "r") as fn:
        for line in fn.readlines():
            text_id, wav_path = line.split()
            valid_at_text_id.append(text_id)

    with open(valid_atflt_fn, "r") as fn:
        for line in fn.readlines():
            text_id, wav_path = line.split()
            valid_atflt_text_id.append(text_id)

    with open(valid_speech_fn, "r") as fn:
        for line in fn.readlines():
            text_id, wav_path = line.split()
            valid_speech_text_id.append(text_id)
    
    with open(valid_nospeech_fn, "r") as fn:
        for line in fn.readlines():
            text_id, wav_path = line.split()
            valid_nospeech_text_id.append(text_id)
    # append
    w_valid_fn = open(valid_fn, "a")
    w_valid_at_fn = open(valid_at_fn, "a")
    w_valid_atflt_fn = open(valid_atflt_fn, "a")
    w_valid_speech_fn = open(valid_speech_fn, "a")
    w_valid_nospeech_fn = open(valid_nospeech_fn, "a")
else:
    # write a new files
    w_valid_fn = open(valid_fn, "w")
    w_valid_at_fn = open(valid_at_fn, "w")
    w_valid_atflt_fn = open(valid_atflt_fn, "w")
    w_valid_speech_fn = open(valid_speech_fn, "w")
    w_valid_nospeech_fn = open(valid_nospeech_fn, "w")

# 已經處理過的id
ignored_text_id = set(valid_text_id)
num_ignored_text_id = len(ignored_text_id)
print(f"{num_ignored_text_id} has been processed")
print("Processing audio")

for i, text_id in tqdm(enumerate(pred_dict.keys())):
    res_len = pred_dict[text_id]
    # 已處理過ID
    if text_id in ignored_text_id:
        continue
    # 異常情況：不是單字詞 (可能有背景人聲，或重複念)
    if res_len > 1:
        continue
    # 異常情況：沒有聲音，但仍然按對
    if res_len == 0 and anno_dict[text_id] == "O":
        continue
    
    wav_path = wav_dict[text_id]
    audio_input_16khz = torch.tensor([ speech_file_to_array_fn(wav_path) ])
    probs = BEATs_model.extract_features(audio_input_16khz)[0]
    
    print(wav_path)
    for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
        top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
        print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')
    input()
