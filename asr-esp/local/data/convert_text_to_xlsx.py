import pandas as pd
import os
import argparse
import json
import re

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="data/icnale/all/whisperv2_large")
parser.add_argument('--corpus_dir', type=str, default="../corpus/speaking/ICNALE")

args = parser.parse_args()

data_dir = args.data_dir
corpus_dir = args.corpus_dir
model_name = data_dir.split("/")[-1]
text_fn = os.path.join(data_dir, "text")
wavscp_fn = os.path.join(data_dir, "../wav.scp")

levels_dict = {"A2_0": 1, "B1_1": 2, "B1_2": 3, "B2_0": 4, "XX_1": 5, "XX_2": 5, "XX_3": 5}

partition_ids = {"train": [], "dev": [], "test":[]}
text_dict = {}
wav_dict = {}

titles = ["id", "wav_path", "spk_id", "holistic", "cefr", "trans_human", "trans_stt"]


for part in list(partition_ids.keys()):
    with open(os.path.join(corpus_dir, "ICNALE_partitions", part + "_ids.tsv")) as fn:
        for line in fn.readlines():
            info = line.split()
            text_id = info[0].split(".")[0]
            partition_ids[part].append(text_id)

with open(text_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_id = info[0]
        transcript = " ".join(info[1:])
        text_dict[text_id] = transcript


with open(wavscp_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_id = info[0]
        wav_path = info[1]
        wav_dict[text_id] = wav_path


xlsx_info = {}
for part in list(partition_ids.keys()):
    xlsx_info[part] = {t:[] for t in titles}
    
    for uttid in list(partition_ids[part]):
        info = uttid.split("_")
        spkid = "_".join(info[:3])
        cefr_level = "_".join(info[-2:])
        holistic = levels_dict[cefr_level]
        trans_stt = re.sub("\[[ A-Za-z]+]", "", text_dict[uttid])
        wav_path = wav_dict[uttid]
        
        xlsx_info[part]["id"].append(uttid)
        xlsx_info[part]["wav_path"].append(wav_path)
        xlsx_info[part]["spk_id"].append(spkid)
        xlsx_info[part]["holistic"].append(holistic)
        xlsx_info[part]["cefr"].append(cefr_level)
        xlsx_info[part]["trans_human"].append("")
        xlsx_info[part]["trans_stt"].append(trans_stt)    
    

xlsx_fn = os.path.join(corpus_dir, "annotations_" + model_name + ".xlsx")

with pd.ExcelWriter(xlsx_fn) as writer:
    for part in list(partition_ids.keys()):
        df = pd.DataFrame(xlsx_info[part])
        df.to_excel(writer, sheet_name=part, index=False)
