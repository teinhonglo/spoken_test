from collections import defaultdict
import numpy as np
import json
import argparse

def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--asr_dir",
                    default="data/train_tr/gigaspeech",
                    type=str)

    args = parser.parse_args()

    asr_dir = args.asr_dir
    fluency_json = {}
    label_json = {}
    
    with open(asr_dir + "/fluency.json", "r") as fn:
        fluency_json = json.load(fn)
    
    with open(asr_dir + "/label.json", "r") as fn:
        label_json = json.load(fn)
    
    uttid_list = list(fluency_json.keys())
    first_uttid = uttid_list[0]
    feats_keys_list = list(fluency_json[first_uttid].keys())
    label_keys_list = list(label_json[first_uttid].keys())
    
    with open(asr_dir + "/data.csv", "w") as fn:
        title_info_list = ["utt_id"] + feats_keys_list + label_keys_list
        title_info = "\t".join(title_info_list)
        fn.write(title_info + "\n")
        
        for uttid in uttid_list:
            utt_info_list = [ uttid ]
            # feats
            for fk in feats_keys_list:
                utt_info_list.append(str(fluency_json[uttid][fk]))
            # label
            for lk in label_keys_list:
                utt_info_list.append(str(label_json[uttid][lk]))
            
            utt_info = "\t".join(utt_info_list)
            fn.write(utt_info + "\n")
