import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--lexicon_fn",
                    default="../models/Librispeech-model-mct-tdnnf/data/local/dict/lexicon.txt",
                    type=str)

parser.add_argument("--text_fn",
                    default="data/spoken_test_2022_jan28/text",
                    type=str)

parser.add_argument("--has_uttid",
                    default=1,
                    type=int)


args = parser.parse_args()

lexicon_fn = args.lexicon_fn
text_fn = args.text_fn

if args.has_uttid == 1:
    has_uttid = True
else:
    has_uttid = False

lexicon_dict = {}
text_dict = {}
oov_counts = defaultdict(int)

with open(lexicon_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        word = info[0]
        prons = " ".join(info[1:])
        lexicon_dict[word] = prons

with open(text_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
            
        if has_uttid:
            words = info[1:]
        else:
            words = info
        
        for word in words:
            if word not in lexicon_dict:
                oov_counts[word] += 1

print(oov_counts)
