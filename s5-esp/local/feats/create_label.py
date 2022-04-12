from collections import defaultdict
import numpy as np
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir",
                    default="data/train_tr/text",
                    type=str)
    
    parser.add_argument("--asr_dir",
                    default="data/train_tr/gigaspeech",
                    type=str)
    
    args = parser.parse_args()
    
    text_fn = args.data_dir + "/text"
    asr_dir = args.asr_dir
    label_dict = {}
    
    with open(text_fn, "r") as fn:
        for line in fn.readlines():
            uttid, fluency_score = line.split()
            label_dict[uttid] = {"label_fluency": fluency_score}
    
    with open(asr_dir + "/label.json", "w") as fn:
        json.dump(label_dict, fn, indent=4)
