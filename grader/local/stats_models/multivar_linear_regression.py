import io
import os
import sys
sys.path.append("./local")
from scipy import stats
import numpy as np
from sklearn import linear_model
from  sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from metrics_cpu import compute_metrics

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="data/gept_b1",
                    type=str)

parser.add_argument("--model_name",
                    default="data/gept_b1/multi_en_mct_cnn_tdnnf_tgt3meg-dl",
                    type=str)

parser.add_argument("--part",
                    default="3",
                    type=str)

parser.add_argument("--aspect",
                    default="2",
                    type=str)

parser.add_argument("--exp_root",
                    default="exp/gept-p2/linear_regression",
                    type=str)

parser.add_argument("--n_resamples",
                    default="-1",
                    type=int)

args = parser.parse_args()

# data/spoken_test_2022_jan28/grader.spk2p3s2
model_name = args.model_name
part = args.part
n_resamples = args.n_resamples
label_fn = "grader.spk2p" + part + "s" + args.aspect
feats_fn = model_name + "-feats.xlsx"

data_dir = args.data_dir

spk2label = {}
spk2feats = {}
aspects_dict = {"1":"content", "2": "pronunciation", "3": "vocabulary"}
n_folds = "1 2 3 4 5".split()

exp_dir = os.path.join(args.exp_root, aspects_dict[args.aspect])

if not os.path.exists(exp_dir):
    print(exp_dir)
    os.makedirs(exp_dir)

def do_resample(X, y, n_resamples=50, scales=[1,2,3,4,5,6,7,8], resample_scales=[1,2,4,6]):
    X_balanced = None
    y_balanced = None    

    print("Resampling")
    for g in scales:
        n_rs = n_resamples
        X_sub = np.copy(X[y==g])
        y_sub = np.copy(y[y==g])
        if len(X_sub) == 0: continue
        
        if g in resample_scales:
            X_copied = None
            while n_rs - len(X_sub) > 0:
                if X_copied is None:
                    X_copied = np.copy(X_sub)
                    y_copied = np.copy(y_sub)
                else:
                    X_copied = np.vstack([X_copied, X_sub])
                    y_copied = np.hstack([y_copied, y_sub])
                n_rs -= len(X_sub)
            
            replace=True
            
            if len(X_sub) >= n_rs:
                replace=False
            
            X_resampled, y_resampled = resample(X_sub,
                                                y_sub,
                                                replace=replace,
                                                n_samples=n_rs,
                                                random_state=66)
            if X_copied is not None:
                X_resampled = np.vstack((X_copied, X_resampled))
                y_resampled = np.hstack((y_copied, y_resampled))
        else:
            X_resampled = X_sub
            y_resampled = y_sub
            
        
        if X_balanced is None or y_balanced is None:
            X_balanced = X_resampled
            y_balanced = y_resampled
        else:
            X_balanced = np.vstack((X_balanced, X_resampled))
            y_balanced = np.hstack((y_balanced, y_resampled))
        print(g, X_sub.shape, y.shape, X_balanced.shape, y_balanced.shape)
    
    return X_balanced, y_balanced

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
    
    kfold_info["All"]["spk_id"] += spk_list.tolist()
    kfold_info["All"]["anno"] += y_test.tolist()
    kfold_info["All"]["anno(cefr)"] += y_test_cefr.tolist()
    kfold_info["All"]["pred"] += y_pred.tolist()
    kfold_info["All"]["pred(cefr)"] += y_pred_cefr.tolist()
    kfold_info["All"]["results"] += (y_pred_cefr - y_test_cefr).tolist()

    
    return acc, macro_avg, weighted_avg, kfold_info

# Feature selection
def feature_selection(X, y, bins):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel
    # https://scikit-learn.org/stable/modules/feature_selection.html
    y_cefr = np.digitize(np.array(y), bins)
    basic_clf = ExtraTreesClassifier(n_estimators=50, random_state=66)
    basic_clf = basic_clf.fit(X, y_cefr)
    importances = basic_clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in basic_clf.estimators_], axis=0)
    selector = SelectFromModel(basic_clf, prefit=True)
    
    return selector, importances, std

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
cefr_bins = np.array([2.5, 4.5, 6.5])
all_bins = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
kf = KFold(n_splits=len(n_folds), random_state=66, shuffle=True)

acc = 0
infos = ["spk_id", "anno", "anno(cefr)", "pred", "pred(cefr)", "results"]
kfold_info = {"Fold" + str(1+i):{info:[] for info in infos} for i in range(len(n_folds))}
kfold_info["All"] = {info:[] for info in infos}
report_titles = ["fold", "acc", "macro_precision", "macro_recall", "macro_f1-score", "weighted_precision", "weighted_recall", "weighted_f1-score"]
report_feats = {"importances":[], "std":[], "feat_keys":[]}
report_dict = {rt: [] for rt in report_titles}
total_losses = {"origin":{}, "cefr":{}}
all_predictions = None
all_annotations = None
all_predictions_cefr = None
all_annotations_cefr = None

# TRAINING (K-FOLD)
for i, (train_index, test_index) in enumerate(kf.split(X)):
    
    kfold_dir = os.path.join(exp_dir, str(i+1))
    
    if not os.path.exists(kfold_dir):
        os.makedirs(kfold_dir)
    
    print("Fold", (i+1))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    if n_resamples != -1:
        X_train, y_train = do_resample(X_train, y_train, n_resamples=n_resamples, resample_scales=[1,2,4,6,8])
    
    selector, importances, std = feature_selection(X_train, y_train, all_bins)
    select_support = selector.get_support() * 1
    select_feat_keys = feat_keys[np.nonzero(select_support)]
    print(select_feat_keys)
    
    plt_importances, plt_std, plt_feat_keys = importances[np.nonzero(select_support)], std[np.nonzero(select_support)], feat_keys[np.nonzero(select_support)]
    report_feats["importances"].append(plt_importances)
    report_feats["std"].append(plt_std)
    report_feats["feat_keys"].append(plt_feat_keys)
    
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
    fold_acc, macro_avg, weighted_avg, kfold_info = report(y_test, y_pred, spk_list[test_index], cefr_bins, kfold_info, "Fold" + str(i+1))
    total_losses["origin"][str(i+1)] = {}
    compute_metrics(total_losses["origin"][str(i+1)], np.array(y_pred), np.array(y_test))
    y_test_cefr = np.digitize(np.array(y_test), cefr_bins)
    y_pred_cefr = np.digitize(np.array(np.round_(y_pred * 2) / 2), cefr_bins)
    total_losses["cefr"][str(i+1)] = {}
    compute_metrics(total_losses["cefr"][str(i+1)], np.array(y_pred_cefr), np.array(y_test_cefr))
    
    report_dict["fold"].append(i+1)
    report_dict["acc"].append(fold_acc)
    report_dict["macro_precision"].append(macro_avg["precision"])
    report_dict["macro_recall"].append(macro_avg["recall"])
    report_dict["macro_f1-score"].append(macro_avg["f1-score"])
    report_dict["weighted_precision"].append(weighted_avg["precision"])
    report_dict["weighted_recall"].append(weighted_avg["recall"])
    report_dict["weighted_f1-score"].append(weighted_avg["f1-score"])
    acc += fold_acc
    
    predictions_file = os.path.join(kfold_dir, "predictions.txt")
    
    with io.open(predictions_file, 'w') as file:
        predictions = '\n'.join(['{} | {}'.format(str(pred), str(target)) for pred, target in zip(y_pred, y_test)])
        file.write(predictions)
    
    if all_predictions is None:
        all_predictions = np.array(y_pred)
        all_annotations = np.array(y_test)
        all_predictions_cefr = y_pred_cefr
        all_annotations_cefr = y_test_cefr
    else:
        all_predictions = np.hstack((all_predictions, np.array(y_pred)))
        all_annotations = np.hstack((all_annotations, np.array(y_test)))
        all_predictions_cefr = np.hstack((all_predictions_cefr, y_pred_cefr))
        all_annotations_cefr = np.hstack((all_annotations_cefr, y_test_cefr))
    
plot_scatter(all_predictions, all_annotations, title="Scatter of predctions and annotations", fig_path=os.path.join(exp_dir, "origin.png"))
plot_scatter(all_predictions_cefr, all_annotations_cefr, title="Scatter of predctions and annotations", fig_path=os.path.join(exp_dir, "cefr.png"))

acc /= kf.get_n_splits(X)
print("Accuracy", acc)

with pd.ExcelWriter(os.path.join(exp_dir, "kfold_detail.xlsx")) as writer:
    for f in list(kfold_info.keys()):
        df = pd.DataFrame(kfold_info[f])
        df.to_excel(writer, sheet_name=f)   

report_df = pd.DataFrame.from_dict(report_dict)
report_df.to_excel(os.path.join(exp_dir, "metric_report.xlsx"), columns=report_titles, index=False)

# visualization
for i in range(len(report_feats["importances"])):
    plt_importances, plt_std, plt_feat_keys = report_feats["importances"][i], report_feats["std"][i], report_feats["feat_keys"][i]
    forest_importances = pd.Series(plt_importances, index=plt_feat_keys).sort_values(ascending=False)
    fig, ax = plt.subplots()
    #forest_importances.plot.bar(yerr=plt_std, ax=ax)
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity (MDI)")
    fig.tight_layout()
    kfold_dir = os.path.join(exp_dir, str(i+1))
    fig.savefig(os.path.join(kfold_dir, "feats-importances_" + str(i+1)+"-fold.png"), dpi=600)


def report_losses(total_losses):
    metrics = list(total_losses[n_folds[0]].keys())
    df_losses = {k:[] for k in list(metrics)}
    
    for nf in n_folds: 
        for metric in list(metrics):
            df_losses[metric].append(total_losses[nf][metric])

    df_losses = pd.DataFrame.from_dict(df_losses)
    print(df_losses.mean())

print("origin")
report_losses(total_losses["origin"])
print("cefr")
report_losses(total_losses["cefr"])
