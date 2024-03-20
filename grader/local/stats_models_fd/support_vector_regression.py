import io
import os
import sys
sys.path.append("./local")
sys.path.append("./local/data")
from scipy import stats
import numpy as np

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.metrics import f1_score
from imblearn import over_sampling, under_sampling

import pandas as pd
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
import argparse

from utils import read_corpus
from metrics_cpu import compute_metrics


parser = argparse.ArgumentParser()

parser.add_argument("--corpus",
                    default="teemi",
                    type=str)

parser.add_argument("--data_dir",
                    default="data-speaking/tb1p1/trans_stt/1",
                    type=str)

parser.add_argument("--feats_path",
                    default="data/pretest/2023_teemi/multi_en_mct_cnn_tdnnf_tgt3meg-dl/all.json",
                    type=str)

parser.add_argument("--score_name",
                    default="pronunciation",
                    type=str)

parser.add_argument("--num_labels",
                    default=8,
                    type=int)                    

parser.add_argument("--exp_dir",
                    default="exp-speaking/tb1p1/linear_regression/pronunciation/1",
                    type=str)

parser.add_argument('--sampling', 
                    default="default", 
                    type=str)

args = parser.parse_args()

corpus = args.corpus
data_dir = args.data_dir
score_name = args.score_name
feats_path = args.feats_path
num_labels = args.num_labels
exp_dir = args.exp_dir
sampling = args.sampling


# Feature selection
def feature_selection(X, y):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    # https://scikit-learn.org/stable/modules/feature_selection.html
    basic_clf = ExtraTreesClassifier(n_estimators=50, random_state=66)
    basic_clf = basic_clf.fit(X, y)
    importances = basic_clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in basic_clf.estimators_], axis=0)
    selector = SelectFromModel(basic_clf, prefit=True)
    
    return selector, importances, std

train_levels, train_info = read_corpus(data_dir + '/train.tsv', num_labels, score_name, corpus)
valid_levels, valid_info = read_corpus(data_dir + '/valid.tsv', num_labels, score_name, corpus)
test_levels, test_info = read_corpus(data_dir + '/test.tsv', num_labels, score_name, corpus)

# prepare feature matrix from feats_path
feats_df = pd.read_excel(feats_path, dtype=str)
feat_keys = [fk for fk in list(feats_df.keys())[6:] if "list" not in fk and "voiced_probs" not in fk]
feat_keys = np.array(feat_keys)
feats_mat_dict = {}

# 在teemi中，text_id在前處理時(抽特徵)會被處理成fname。
for i, text_id in enumerate(feats_df["fname"]):
    if text_id in feats_mat_dict:
        print("[Prepare Feats] The ID {} shouldn't be unique. Remember to recheck the feature file {} .".format(text_id, feats_path))
        exit(0)
    feats_vec = [float(feats_df[fk][i]) for fk in feat_keys]
    feats_mat_dict[text_id]  = feats_vec

train_feats = [ feats_mat_dict[text_id] for text_id in train_info["ids"]]
valid_feats = [ feats_mat_dict[text_id] for text_id in valid_info["ids"]]
test_feats = [ feats_mat_dict[text_id] for text_id in test_info["ids"]]

# NOTE: valid == test
X_train, X_test = train_feats, valid_feats
y_train, y_test = train_levels, valid_levels

# feature selection
#selector, importances, std = feature_selection(X_train, y_train)
#select_support = selector.get_support() * 1
#select_feat_keys = feat_keys[np.nonzero(select_support)]

#X_train = selector.transform(X_train)
#X_test = selector.transform(X_test)
if sampling == "smote":
    X_train, y_train = over_sampling.SMOTE(random_state=66).fit_resample(X_train, y_train) 
elif sampling == "bsmote":
    X_train, y_train = over_sampling.BorderlineSMOTE(random_state=66, kind='borderline-2').fit_resample(X_train, y_train) 
elif sampling == "tomek_link":
    X_train, y_train = under_sampling.TomekLinks().fit_resample(X_train, y_train) 

# model training
model = svm.SVR(random_state=66)
model.fit(X_train, y_train)

model_dir = os.path.join(exp_dir, "checkpoint")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

y_pred = model.predict(X_test)
 
#coef_ = model.coef_[np.nonzero(model.coef_)]
#feat_nz_keys = select_feat_keys[np.nonzero(model.coef_)]
    
#print("=" * 10, "Feature Importance", "=" * 10)
#print(feat_nz_keys[np.argsort(-1 * coef_)])
#print(coef_[np.argsort(-1 * coef_)])
    

gold_labels, pred_labels = y_test.tolist(), y_pred.tolist()
total_losses = {}
compute_metrics(total_losses, y_test, y_pred)

model_name = "MLR-mcrmse={}.ckpt".format(float(total_losses["mcrmse"]))
joblib.dump(model, os.path.join(model_dir, model_name))
model = joblib.load(os.path.join(model_dir, model_name))

log_dir = exp_dir
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(os.path.join(log_dir, "predictions.txt"), "w") as file:
    predictions_info = '\n'.join(['{} {}'.format(str(pred), str(target)) for pred, target in zip(pred_labels, gold_labels)])
    file.write(predictions_info)
