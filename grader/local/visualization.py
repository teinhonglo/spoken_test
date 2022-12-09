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

args = parser.parse_args()

result_root = args.result_root
plot_sheets = args.plot_sheets.split()
scores = args.scores.split()
anno_columns = ["anno", "anno(cefr)"]
pred_columns = ["pred", "pred(cefr)"]
read_columns = anno_columns + pred_columns

kfold_info = {}
all_bins = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
cefr_bins = np.array([1.5, 2.5, 3.5])

for score in scores:
    kfold_info[score] = defaultdict(dict)
    for nf in plot_sheets:
        xlsx_path = os.path.join(result_root, score, "kfold_detail.xlsx")
        df = pd.read_excel(xlsx_path, sheet_name=nf)
        for rc in read_columns:
            kfold_info[score][nf][rc] = df[rc]


# plot confusion matrix
for score in list(kfold_info.keys()):
    for nf in list(kfold_info[score].keys()):
        for anno_type, pred_type in zip(anno_columns, pred_columns):
            file_name = os.path.join(result_root, "-".join([score, pred_type, nf])) + ".png"
             
            y_true = kfold_info[score][nf][anno_type]
            y_pred = kfold_info[score][nf][pred_type]
            if anno_type == "anno":
                y_true = np.digitize(y_true, all_bins)
                y_pred = np.digitize(y_pred, all_bins)
                labels = ["pre-A","A1","A1A2","A2","A2B1","B1","B1B2", "B2"]
            else:
                y_true = np.digitize(y_true, cefr_bins)
                y_pred = np.digitize(y_pred, cefr_bins)
                labels = ["A1","A2","B1", "B2"]
            
            conf_mat = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
            conf_mat_df = pd.DataFrame(conf_mat, index=labels, columns=labels)
                        
            #print(conf_mat)
            sns.heatmap(conf_mat_df, annot=True, fmt='g')
            plt.xlabel("Predictions")
            plt.ylabel("Annotations")
            plt.savefig(file_name)
            plt.clf()