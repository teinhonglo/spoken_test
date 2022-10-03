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

def get_stats(numeric_list, prefix=""):
    # number, mean, standard deviation (std), median, mean absolute deviation
    stats_np = np.array(numeric_list)
    number = len(stats_np)
    
    if number == 0:
        summ = 0.
        mean = 0.
        std = 0.
        median = 0.
        mad = 0.
        maximum = 0.
        minimum = 0.
    else:
        summ = np.sum(stats_np)
        mean = np.mean(stats_np)
        std = np.std(stats_np)
        median = np.median(stats_np)
        mad = np.sum(np.absolute(stats_np - mean)) / number
        maximum = np.max(stats_np)
        minimum = np.min(stats_np)
    
    stats_dict = {  prefix + "number": number, 
                    prefix + "mean": mean, 
                    prefix + "std": std, 
                    prefix + "median": median, 
                    prefix + "mad": mad, 
                    prefix + "summ": summ,
                    prefix + "max": maximum,
                    prefix + "min": minimum
                 }
    return stats_dict

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


args = parser.parse_args()

# data/spoken_test_2022_jan28/grader.spk2p3s2
model_name = args.model_name
part = args.part
text_fn = "text." + part + "p"
feats_fn = model_name + "-feats.xlsx"

data_dir = args.data_dir

word_count = {}

# label
with open(os.path.join(data_dir, text_fn), "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        words = info[1:]
        word_count[uttid] = len(words)
        
print(get_stats(list(word_count.values())))
