import os
import json
import soundfile
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--trans_root",
                    default="/share/corpus/2023_teemiv2/transcript",
                    type=str)

parser.add_argument("--trans_fns",
                    default="",
                    type=str)

parser.add_argument("--data_dir",
                    default="data/2023_teemi_stt",
                    type=str)

args = parser.parse_args()

trans_root = args.trans_root
trans_fns = args.trans_fns.split(",")
data_dir = args.data_dir

text_dict = {}

for trans_fn in trans_fns:
    anno_df = pd.read_excel(os.path.join(trans_root))
    for i in range(len())
