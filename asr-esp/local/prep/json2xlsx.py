import pandas as pd
import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="data/voice_2022/gigaspeech")

args = parser.parse_args()

data_dir = args.data_dir
model_name = data_dir.split("/")[-1]
json_fn = os.path.join(data_dir, "all.json")

titles = ["fname", "spkID", "part", "qID", "prompt", "stt"]

with open(json_fn, "r") as fn:
    all_json = json.load(fn)

xlsx_info = {t:[] for t in titles}

first_spk = list(all_json.keys())[0]
feats_type = list(all_json[first_spk]["feats"].keys())

for f in feats_type:
    titles.append(f)

for uttid in list(all_json.keys()):
    utt_info = all_json[uttid]
    wav_path = utt_info["wav_path"]
    fname = wav_path.split("/")[-1].split(".")[0]
    part = uttid.split("-")[1]
    qID = "-".join(uttid.split("-")[1:3])
    spkID = uttid.split("-")[0]
    
    xlsx_info["fname"].append(fname)
    xlsx_info["spkID"].append(spkID)
    xlsx_info["part"].append(part)
    xlsx_info["qID"].append(qID)
    xlsx_info["prompt"].append(utt_info["prompt"])
    xlsx_info["stt"].append(utt_info["stt"])
    
    feats_info = utt_info["feats"]
    
    for key, val in feats_info.items():
        if key in xlsx_info:
            xlsx_info[key].append(val)
        else:
            xlsx_info[key] = [val]


df = pd.DataFrame(xlsx_info)
df.to_excel(os.path.join(data_dir, "all.xlsx"), index=False)
