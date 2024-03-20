import os
import json
import soundfile
from tqdm import tqdm
import numpy as np
import sys
import wave
import whisper
from whisper.normalizers import EnglishTextNormalizer
import string
import jiwer

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--ref",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56/text",
                    type=str)

parser.add_argument("--hyp",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56/text",
                    type=str)


args = parser.parse_args()


ref = args.ref
hyp = args.hyp
ref_dict = {}
hyp_dict = {}
utt_list = []
normalizer = EnglishTextNormalizer()

with open(ref, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        ref_dict[info[0]] = normalizer(" ".join(info[1:]))
        utt_list.append(info[0])

with open(hyp, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        hyp_dict[info[0]] = normalizer(" ".join(info[1:]))


print("WER(%)")
data = {}
data["hypothesis_clean"] = []
data["reference_clean"] = []

for uttid in utt_list:
    if ref_dict[uttid] == "": continue
    data["reference_clean"].append(ref_dict[uttid].upper())
    data["hypothesis_clean"].append(hyp_dict[uttid].upper())

wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
print(f"WER: {wer * 100:.2f} %")
    
