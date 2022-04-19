import os
import numpy as np
import json
import soundfile
from collections import defaultdict
from tqdm import tqdm
'''
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
'''
def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

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
    
    
class SpeechModel(object):
    def __init__(self, recog_dict, gop_result_dir, gop_json_fn):
        # STT
        self.recog_dict = recog_dict
        self.gop_ctm_info = self.get_gop_ctm(gop_result_dir, gop_json_fn)
        # Fluency
        self.sil_seconds = 0.145
        self.long_sil_seconds = 0.495
        self.disflunecy_words = ["AH", "UM", "UH", "EM"]
        self.special_words = ["<UNK>"]
    
    # STT features
    def recog(self, uttid):
        text = self.recog_dict[uttid]
        return text
    
    def get_ctm(self, uttid):
        ctm_info = self.gop_ctm_info[uttid]
        return ctm_info
    
    def get_gop_ctm(self, gop_result_dir, gop_json_fn):
        # confidence (GOP)
        with open(gop_json_fn, "r") as fn:
            gop_json = json.load(fn)
        
        gop_ctm_info = defaultdict(list)
        
        # word-level ctm
        with open(os.path.join(gop_result_dir, "word.ctm")) as wctm_fn:
            count = 0
            prev_uttid = None
            for line in wctm_fn.readlines():
                uttid, _, start_time, duration, word_id = line.split()
                if prev_uttid != uttid:
                    count = 0
                # NOTE: 
                word_gop_id, word_gop_info = gop_json[uttid]["GOP"][count]
                word_gop = word_gop_info[-1][-1]
                
                assert word_id == word_gop_id
                
                start_time = round(float(start_time), 4)
                duration = round(float(duration), 4)
                conf = round(float(word_gop) / 100, 4)
                
                if conf > 1.0:
                    conf = 1.0
                if conf < 0.0:
                    conf = 0.0
                
                gop_ctm_info[uttid].append([word_id, start_time, duration, conf])
                count += 1
                prev_uttid = uttid
        
        return gop_ctm_info
    
    # Fluency features
    def sil_feats(self, ctm_info, response_duration):
        # > 0.145
        sil_list = []
        # > 0.495
        long_sil_list = []
        if len(ctm_info) > 2:
            word, start_time, duration, conf = ctm_info[0]
            end_time = start_time + duration
            
            for word, start_time, duration, conf in ctm_info[1:]:
                interval_word_duration = start_time - end_time
                
                if interval_word_duration > self.sil_seconds:
                    sil_list.append(interval_word_duration)
                
                if interval_word_duration > self.long_sil_seconds:
                    long_sil_list.append(interval_word_duration)
                
                end_time = start_time + duration
        
        
        sil_stats = get_stats(sil_list, prefix="sil_")
        long_sil_stats = get_stats(long_sil_list, prefix="long_sil_")
        '''
        {sil, long_sil}_rate1: num_silences / response_duration
        {sil, long_sil}_rate2: num_silences / num_words
        '''
        num_sils = len(sil_list)
        num_long_sils = len(long_sil_list)
        num_words = len(ctm_info)
        
        sil_stats["sil_rate1"] = num_sils / response_duration
        
        if num_words > 0:
            sil_stats["sil_rate2"] = num_sils / num_words
        else:
            sil_stats["sil_rate2"] = 0
        
        long_sil_stats["long_sil_rate1"] = num_long_sils / response_duration 
        
        if num_words > 0:
            long_sil_stats["long_sil_rate2"] = num_long_sils / num_words
        else:
            long_sil_stats["long_sil_rate2"] = 0
        
        sil_dict = merge_dict(sil_stats, long_sil_stats)
        
        return sil_dict
    
    def word_feats(self, ctm_info, response_duration):
        '''
        TODO:
        number of repeated words
        '''
        word_count = len(ctm_info)
        word_duration_list = []
        word_conf_list = []
        num_disfluecy = 0
        
        for word, start_time, duration, conf in ctm_info:
            word_duration_list.append(duration)
            word_conf_list.append(conf)
            if word in self.disflunecy_words:
                num_disfluecy += 1
            
        # strat_time and duration of last word
        # word in articlulation time
        word_freq = word_count / response_duration
        word_duration_stats = get_stats(word_duration_list, prefix = "word_duration_")
        word_conf_stats = get_stats(word_conf_list, prefix="word_conf_")
        
        word_basic_dict = {   
                        "word_count": word_count,
                        "word_freq": word_freq,
                        "word_num_disfluency": num_disfluecy
                    }
        word_stats_dict = merge_dict(word_duration_stats, word_conf_stats)
        word_dict = merge_dict(word_basic_dict, word_stats_dict)
        
        return word_dict
     
