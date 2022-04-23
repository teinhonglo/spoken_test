import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

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

spk2label = {}
spk2feats = {}


# label
with open(os.path.join(data_dir, label_fn), "r") as fn:
    for line in fn.readlines():
        spk, grade = line.split()
        spk2label[spk] = float(grade)

# feats
feats_df = pd.read_excel(os.path.join(data_dir, model_name, feats_fn), dtype=str)
feat_keys = [fk for fk in list(feats_df.keys())[6:] if "list" not in fk and "voiced_probs" not in fk]

for i, spk in enumerate(feats_df["spkID"]):
    if feats_df["part"][i] != part: continue
    
    feats_vec = [float(feats_df[fk][i]) for fk in feat_keys]
    spk2feats[spk] = feats_vec

# create example
X, y, spk_list = [], [], []
for spk in list(spk2label.keys()):
    X.append(spk2feats[spk])
    y.append(spk2label[spk])
    spk_list.append(spk)

X = np.array(X)
y = np.array(y)

m = len(y) # Number of training examples
b1_bins = np.array([4.0, 5.0])
all_bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
kf = KFold(n_splits=5, random_state=66, shuffle=True)

def report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print("MSE", mean_squared_error(y_true, y_pred))
    print("RMSE", mean_squared_error(y_true, y_pred, squared=False))

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print("Fold", i)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    clf = GradientBoostingRegressor(random_state=66, n_estimators=250)
    clf.fit(X_train, y_train)
    
    # print(b1_bins) 
    y_pred = clf.predict(X_test)
    print(y_test)
    print(y_pred)
    y_test_cefr = np.digitize(np.array(y_test), b1_bins)
    y_pred_cefr = np.digitize(np.array(np.round_(y_pred * 2)/2), b1_bins)
    print(y_test_cefr)
    print(y_pred_cefr)
    report(y_test_cefr, y_pred_cefr) 
    print()
   
