import numpy as np
import pandas as pd
from sklearn import linear_model
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from  sklearn import preprocessing
from scipy import stats

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data/spoken_test_2022_jan28",
                    type=str)

parser.add_argument("--model_name",
                    default="librispeech_mct_tdnnf_kaldi_tgt3",
                    type=str)

parser.add_argument("--part",
                    default="3",
                    type=str)

parser.add_argument("--aspect",
                    default="2",
                    type=str)

args = parser.parse_args()

# data/spoken_test_2022_jan28/grader.spk2p3s2
model_name = args.model_name
part = args.part
label_fn = "grader.spk2p" + part + "s" + args.aspect
feats_fn = model_name + "-feats.xlsx"

data_dir = args.data_dir
result_root = "../automated-english-transcription-grader/data"

spk2label = {}
spk2feats = {}

# label
with open(os.path.join(data_dir, label_fn), "r") as fn:
    for line in fn.readlines():
        spk, grade = line.split()
        spk2label[spk] = float(grade)

# feats
feats_df = pd.read_excel(os.path.join(data_dir, model_name, feats_fn), dtype=str)
#feat_keys = [fk for fk in list(feats_df.keys())[6:] if "list" not in fk and "voiced_probs" not in fk]
feats_map = {"stt": "text", "prompt": "prompt"}
feats_keys = list(feats_map.keys())
ordered_headers = ["fname", "spkID"]

for f in feats_keys:
    ordered_headers.append(f)

ordered_headers.append("cefr(ori)")
ordered_headers.append("score")

feats_df = feats_df.loc[feats_df["part"] == part]
feats_df.reset_index(drop=True, inplace=True)
new_feats_dict = {k:[] for k in ordered_headers}

for i, spk in enumerate(feats_df["spkID"]):
    for k in ordered_headers[:-2]:
        new_feats_dict[k].append(feats_df[k][i])
    
    new_feats_dict["cefr(ori)"].append(spk2label[spk])
    new_feats_dict["score"].append(spk2label[spk])


b1_bins = np.array([1.0, 4.0, 5.0])
all_bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
new_feats_dict["cefr(ori)"] = np.digitize(new_feats_dict["cefr(ori)"], all_bins)
new_feats_dict["score"] = np.digitize(new_feats_dict["score"], b1_bins)

new_feats_df = pd.DataFrame.from_dict(new_feats_dict)

kf = KFold(n_splits=5, random_state=66, shuffle=True)

# TRAINING (K-FOLD)
for i, (train_index, test_index) in enumerate(kf.split(new_feats_df)):
    kfold_dir = "Fold" + str(i+1)
    result_dir = os.path.join(result_root, kfold_dir)
    
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    train_df, test_df = new_feats_df.iloc[train_index], new_feats_df.iloc[test_index] 
    
    train_df.to_csv(os.path.join(result_dir, "train.tsv"), header=ordered_headers, sep="\t")
    test_df.to_csv(os.path.join(result_dir, "valid.tsv"), header=ordered_headers, sep="\t")
    test_df.to_csv(os.path.join(result_dir, "test.tsv"), header=ordered_headers, sep="\t")
