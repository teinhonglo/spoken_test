from collections import defaultdict
import numpy as np
import json
import argparse

def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

class FluencyModel(object):
    def __init__(self, all_json):
        # preprocess
        self.all_json = all_json
        self.long_sil_seconds = 0.495
        self.sil_seconds = 0.145
        self.disflunecy_words = ["AH", "UM"]
    
    def get_uttid(self):
        return list(self.all_json.keys())
    
    def get_stats(self, stats_list):
        # number, mean, standard deviation (std), median, mean absolute deviation
        stats_np = np.array(stats_list)
        number = len(stats_list)
        
        if number == 0:
            summ = 0.
            mean = 0.
            std = 0.
            median = 0.
            mad = 0.
        else:
            summ = np.sum(stats_np)
            mean = np.mean(stats_np)
            std = np.std(stats_np)
            median = np.median(stats_np)
            mad = np.sum(np.absolute(stats_np - mean)) / number
        
        return {"number": number, "mean": mean, "std": std, "median": median, "mad": mad, "summ": summ}        
    
    def sil_feats(self, utt_id):
        # > 0.145
        sil_list = []
        # > 0.495
        long_sil_list = []
        text, start_time, duration, conf = self.all_json[utt_id]["ctm"][0]
        end_time = start_time + duration
        
        for word, start_time, duration, conf in self.all_json[utt_id]["ctm"][1:]:
            interval_word_duration = start_time - end_time
            
            if interval_word_duration > self.sil_seconds:
                sil_list.append(interval_word_duration)
            
            if interval_word_duration > self.long_sil_seconds:
                long_sil_list.append(interval_word_duration)
            
            end_time = start_time + duration
        
        sil_stats = {"sil_" + k: v for k, v in self.get_stats(sil_list).items()}
        long_sil_stats = {"long_sil_" + k:v for k, v in self.get_stats(long_sil_list).items()}
        
        sil_dict = merge_dict(sil_stats, long_sil_stats)
        return sil_dict
    
    
    def word_feats(self, utt_id):
        word_count = len(self.all_json[utt_id]["ctm"])
        word_duration_list = []
        word_conf_list = []
        num_disfluecy = 0
        
        for word, start_time, duration, conf in self.all_json[utt_id]["ctm"]:
            word_duration_list.append(duration)
            word_conf_list.append(np.exp(conf))
            if word in self.disflunecy_words:
                num_disfluecy += 1
            
        # strat_time and duration of last word
        word_freq = word_count / (start_time + duration)
        word_duration_stats = self.get_stats(word_duration_list)
        word_conf_stats = self.get_stats(word_conf_list)
        mean_duration = word_duration_stats["mean"]
        
        word_dict = {   "word_count": word_count, 
                        "word_freq": word_freq, 
                        "word_mean_duration": mean_duration, 
                        "word_confidence_mean": word_conf_stats["mean"], 
                        "word_num_disfluency": num_disfluecy
                    }
        return word_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--asr_dir",
                    default="data/train_tr/gigaspeech",
                    type=str)

    args = parser.parse_args()

    asr_dir = args.asr_dir
    all_json = {}
    
    with open(asr_dir + "/all.json", "r") as fn:
        all_json = json.load(fn)

    fluency_model = FluencyModel(all_json)
    utt_list = fluency_model.get_uttid()
    fluency_feats_dict = {}
    
    for uttid in utt_list:
        sil_dict = fluency_model.sil_feats(uttid)
        word_dict = fluency_model.word_feats(uttid)
        fluency_feats_dict[uttid] = merge_dict(sil_dict, word_dict)

    with open(asr_dir + "/fluency.json", "w") as fn:
        json.dump(fluency_feats_dict, fn, indent=4)
