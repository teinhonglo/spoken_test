import argparse
import random
import logging
import os
import csv
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
from metrics_cpu import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument("--result_root",
                    default="runs-speaking/gept-p3/trans_stt_tov_round/bert-model",
                    type=str)

parser.add_argument("--plot_sheets",
                    default="All",
                    type=str)
                    
parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--all_bins",
                    default="1.5,2.5,3.5,4.5,5.5,6.5,7.5",
                    type=str)

parser.add_argument("--cefr_bins",
                    default="2.5,4.5,6.5",
                    type=str)

parser.add_argument("--affix",
                    default="",
                    type=str)

args = parser.parse_args()

result_root = args.result_root
plot_sheets = args.plot_sheets.split()
scores = args.scores.split()
affix = args.affix

anno_columns = ["anno", "anno(cefr)"]
pred_columns = ["pred", "pred(cefr)"]
read_columns = anno_columns + pred_columns

kfold_info = {}
all_bins = np.array([ float(ab) for ab in args.all_bins.split(",")])
cefr_bins = np.array([ float(cb) for cb in args.cefr_bins.split(",")])

for score in scores:
    kfold_info[score] = defaultdict(dict)
    for nf in plot_sheets:
        xlsx_path = os.path.join(result_root, score, "kfold_detail" + affix + ".xlsx")
        df = pd.read_excel(xlsx_path, sheet_name=nf)
        for rc in read_columns:
            kfold_info[score][nf][rc] = df[rc]


# plot confusion matrix
for score in list(kfold_info.keys()):
    for nf in list(kfold_info[score].keys()):
        for anno_type, pred_type in zip(anno_columns, pred_columns):
            file_name = os.path.join(result_root, score, "-".join([score, pred_type, nf]) + affix)
            png_name = file_name + ".png"
            excel_name = file_name + ".xlsx"
             
            y_true = kfold_info[score][nf][anno_type]
            y_pred = kfold_info[score][nf][pred_type]
            if anno_type == "anno":
                y_true = y_true - 1 #np.digitize(y_true, all_bins)
                y_pred = y_pred - 1 #np.digitize(y_pred, all_bins)
                labels = ["pre-A","A1","A1A2","A2","A2B1","B1","B1B2", "B2"]
            else:
                y_true = y_true - 1 #np.digitize(y_true, cefr_bins)
                y_pred = y_pred - 1 #np.digitize(y_pred, cefr_bins)
                labels = ["A1","A2","B1", "B2"]
            
            conf_mat = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
            row_sum = np.sum(conf_mat, axis = 1)
            conf_mat_prec = conf_mat / row_sum[:, np.newaxis]
            conf_mat_prec[np.where(conf_mat == 0)] = 0
            conf_mat_df = pd.DataFrame(conf_mat, index=labels, columns=labels)
            conf_mat_prec_df = pd.DataFrame(conf_mat_prec, index=labels, columns=labels)

            #print(conf_mat)
            sns.heatmap(data=conf_mat_prec_df, annot=conf_mat_df, fmt='g')
            plt.xlabel("Predictions")
            plt.ylabel("Annotations")
            plt.savefig(png_name)
            plt.clf()
            
            #conf_mat_prec_df.to_excel(excel_name)
