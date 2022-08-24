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
import logging

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

parser.add_argument("--exp_dir",
                    default="exp/linear_regression",
                    type=str)

args = parser.parse_args()

# data/spoken_test_2022_jan28/grader.spk2p3s2
model_name = args.model_name
part = args.part
label_fn = "grader.spk2p" + part + "s" + args.aspect
feats_fn = model_name + "-feats.xlsx"

data_dir = args.data_dir
exp_dir = args.exp_dir

if not os.path.exists(exp_dir):
    os.path.mkdirs(exp_dir)

spk2label = {}
spk2feats = {}

def report(y_test, y_pred, spk_list, bins, kfold_info, fold="Fold1"):
    print("=" * 10, "Raw data", "=" * 10)
    y_test_cefr = np.digitize(np.array(y_test), bins)
    y_pred_cefr = np.digitize(np.array(np.round_(y_pred * 2) / 2), bins)
    print("spk_list, y_test, y_test_cefr, y_pred, y_pred_cefr")
    for i in range(len(spk_list)):
        print(spk_list[i], y_test[i], y_test_cefr[i], y_pred[i], y_pred_cefr[i])
    print("="* 10)
    
    print("="*10, "Coefficient", "=" * 10)
    print("MSE", mean_squared_error(y_test, y_pred))
    print("Pearson")
    
    df = pd.DataFrame({"anno": y_test,
                       "pred": y_pred})
    
    print(df.corr(method='pearson'))
    print(stats.pearsonr(y_test, y_pred))
     
    print("="*10, "Classification", "=" * 10)
    print(classification_report(y_test_cefr, y_pred_cefr))
    print(confusion_matrix(y_test_cefr, y_pred_cefr))
    print()
    classfication_report_detail = classification_report(y_test_cefr, y_pred_cefr, output_dict=True)
    acc = classfication_report_detail["accuracy"]
    macro_avg = classfication_report_detail["macro avg"]
    weighted_avg = classfication_report_detail["weighted avg"]
    
    kfold_info[fold]["spk_id"] += spk_list.tolist()
    kfold_info[fold]["anno"] += y_test.tolist()
    kfold_info[fold]["anno(cefr)"] += y_test_cefr.tolist()
    kfold_info[fold]["pred"] += y_pred.tolist()
    kfold_info[fold]["pred(cefr)"] += y_pred_cefr.tolist()
    kfold_info[fold]["results"] += (y_pred_cefr - y_test_cefr).tolist()
    
    return acc, macro_avg, weighted_avg, kfold_info

# Feature selection
def feature_selection(X, y, bins):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    # https://scikit-learn.org/stable/modules/feature_selection.html
    y_cefr = np.digitize(np.array(y), bins)
    basic_clf = ExtraTreesClassifier(n_estimators=50, random_state=66)
    basic_clf = basic_clf.fit(X, y_cefr)
    selector = SelectFromModel(basic_clf, prefit=True)
    
    return selector

# label
with open(os.path.join(data_dir, label_fn), "r") as fn:
    for line in fn.readlines():
        spk, grade = line.split()
        spk2label[spk] = float(grade)

# feats
feats_df = pd.read_excel(os.path.join(data_dir, model_name, feats_fn), dtype=str)
feat_keys = [fk for fk in list(feats_df.keys())[6:] if "list" not in fk and "voiced_probs" not in fk]
feat_keys = np.array(feat_keys)

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
spk_list = np.array(spk_list)

m = len(y) # Number of training examples
b1_bins = np.array([1.0, 4.0, 5.0])
all_bins = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
kf = KFold(n_splits=5, random_state=66, shuffle=True)

acc = 0
infos = ["spk_id", "anno", "anno(cefr)", "pred", "pred(cefr)", "results"]
kfold_info = {"Fold" + str(1+i):{info:[] for info in infos} for i in range(5)}
report_titles = ["fold", "acc", "macro_precision", "macro_recall", "macro_f1-score", "weighted_precision", "weighted_recall", "weighted_f1-score"]
report_dict = {rt: [] for rt in report_titles}

# TRAINING (K-FOLD)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    print("Fold", (i+1))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    selector = feature_selection(X_train, y_train, b1_bins)
    select_support = selector.get_support() * 1
    select_feat_keys = feat_keys[np.nonzero(select_support)]
    print(select_feat_keys)
    
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)
    
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train, y_train)
    
    coef_ = clf.coef_[np.nonzero(clf.coef_)]
    feat_nz_keys = select_feat_keys[np.nonzero(clf.coef_)]
    
    print("=" * 10, "Feature Importance", "=" * 10)
    print(feat_nz_keys[np.argsort(-1 * coef_)])
    print(coef_[np.argsort(-1 * coef_)])
    
    y_pred = clf.predict(X_test) 
    fold_acc, macro_avg, weighted_avg, kfold_info = report(y_test, y_pred, spk_list[test_index], b1_bins, kfold_info, "Fold" + str(i+1))
    
    report_dict["fold"].append(i)
    report_dict["acc"].append(fold_acc)
    report_dict["macro_precision"].append(macro_avg["precision"])
    report_dict["macro_recall"].append(macro_avg["recall"])
    report_dict["macro_f1-score"].append(macro_avg["f1-score"])
    report_dict["weighted_precision"].append(weighted_avg["precision"])
    report_dict["weighted_recall"].append(weighted_avg["recall"])
    report_dict["weighted_f1-score"].append(weighted_avg["f1-score"])
    
    acc += fold_acc
    
acc /= kf.get_n_splits(X)
print("Accuracy", acc)

with pd.ExcelWriter(os.path.join(exp_dir, "linear_regression.xlsx")) as writer:
    for f in list(kfold_info.keys()):
        df = pd.DataFrame(kfold_info[f])
        df.to_excel(writer, sheet_name=f)   

report_df = pd.DataFrame.from_dict(report_dict)
report_df.to_excel(os.path.join(exp_dir, "linear_regression_report.xlsx"), columns=report_titles, index=False)
