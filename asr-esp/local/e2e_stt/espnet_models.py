from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.asr_align import CTCSegmentation

import os
import numpy as np
import json
import soundfile
from collections import defaultdict
from tqdm import tqdm
from g2p_en import G2p


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
    def __init__(self, tag, is_download=True, cache_dir="./downloads"):
        # STT
        if is_download:
            d=ModelDownloader(cachedir=cache_dir)
            asr_model = d.download_and_unpack(tag)
            self.speech2text = Speech2Text.from_pretrained(
                **asr_model,
                device="cpu",
                maxlenratio=0.0,
                minlenratio=0.0,
                beam_size=20,
                ctc_weight=0.3,
                lm_weight=0.3,
                penalty=0.0,
                nbest=1
            )
            self.aligner = CTCSegmentation(**asr_model, fs=16000, ngpu=0, kaldi_style_text=False, time_stamps="auto")
        # Fluency
        self.sil_seconds = 0.145
        self.long_sil_seconds = 0.495
        self.disflunecy_words = ["AH", "UM", "UH", "EM", "OH"]
        self.special_words = ["<UNK>"]
        self.g2p = G2p()
    
    # STT features
    def recog(self, speech):
        nbests = self.speech2text(speech)
        text, *_ = nbests[0]
        #text = self.asr_text_post_processing(text)
        return text
    
    def asr_text_post_processing(self, text):
        # 1. convert to uppercase
        text = text.upper()
        
        # 2. remove hyphen
        #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
        text = text.replace('-', ' ')
        
        # 3. remove non-scoring words from evaluation
        remaining_words = []
        
        for word in text.split():
            if word in non_scoring_words:
                continue
            remaining_words.append(word)
        
        return ' '.join(remaining_words)

    
    def get_ctm(self, speech, text):
        # alignment (stt)
        segments = self.aligner(speech, text.split())
        segment_info = segments.segments
        text_info = segments.text
        ctm_info = []
        
        for i in range(len(segment_info)):
            start_time, end_time, conf = segment_info[i]
            start_time = round(start_time, 4)
            end_time = round(end_time, 4)
            duration = round(end_time - start_time, 4)
            conf = round(conf, 4)
            ctm_info.append([text_info[i], start_time, duration, round(np.exp(conf),4)])
        
        return ctm_info
    
    def get_phone_ctm(self, ctm_info):
        # use g2p model
        phone_ctm_info = []
        phone_text = []
        
        for word, start_time, duration, conf in ctm_info:
            phones = self.g2p(word)
            duration /= len(phones)
            
            for phone in phones:
                phone_ctm_info.append([phone, start_time, duration, conf])
                start_time += duration
                phone_text.append(phone)
        
        phone_text = " ".join(phone_text)
        
        return phone_ctm_info, phone_text
    
    # Fluency features
    def sil_feats(self, ctm_info, total_duration):
        # > 0.145
        sil_list = []
        # > 0.495
        long_sil_list = []
        
        response_duration = total_duration
        if len(ctm_info) > 0:
            # response time
            start_time = ctm_info[0][1]
            # start_time + duration
            end_time = ctm_info[-1][1] + ctm_info[-1][2]
            response_duration = end_time - start_time        

        # word-interval silence
        if len(ctm_info) > 2:
            word, start_time, duration, conf = ctm_info[0]
            prev_end_time = start_time + duration
            
            for word, start_time, duration, conf in ctm_info[1:]:
                interval_word_duration = start_time - prev_end_time
                
                if interval_word_duration > self.sil_seconds:
                    sil_list.append(interval_word_duration)
                
                if interval_word_duration > self.long_sil_seconds:
                    long_sil_list.append(interval_word_duration)
                
                prev_end_time = start_time + duration
        
        
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
        
        return sil_dict, response_duration
    
    def word_feats(self, ctm_info, total_duration):
        '''
        TODO:
        number of repeated words (short pauses)
        distinct word
        articulation rate
        '''
        word_count_dict = defaultdict(int)
        word_duration_list = []
        word_conf_list = []
        num_disfluecy = 0
        num_repeat = 0
        prev_words = []
        
        response_duration = total_duration
        if len(ctm_info) > 0:
            # response time
            start_time = ctm_info[0][1]
            # start_time + duration
            end_time = ctm_info[-1][1] + ctm_info[-1][2]
            response_duration = end_time - start_time        
        
        for word, start_time, duration, conf in ctm_info:
            word_duration_list.append(duration)
            word_conf_list.append(conf)
            word_count_dict[word] += 1
            
            if word in self.disflunecy_words:
                num_disfluecy += 1
            
            if word in prev_words:
                num_repeat += 1
            
            prev_words = [word]
            
        # strat_time and duration of last word
        # word in articlulation time
        word_count = sum(list(word_count_dict.values()))
        word_distinct = len(list(word_count_dict.keys()))
        word_freq = word_count / response_duration
        word_duration_stats = get_stats(word_duration_list, prefix = "word_duration_")
        word_conf_stats = get_stats(word_conf_list, prefix="word_conf_")
        
        word_basic_dict = { 
                            "word_count": word_count,
                            "word_distinct": word_distinct,
                            "word_freq": word_freq,
                            "word_num_disfluency": num_disfluecy,
                            "word_num_repeat": num_repeat
                          }
        
        word_stats_dict = merge_dict(word_duration_stats, word_conf_stats)
        word_dict = merge_dict(word_basic_dict, word_stats_dict)
        
        return word_dict, response_duration
     
    def phone_feats(self, ctm_info, total_duration):
        phone_count_dict = defaultdict(int)
        phone_duration_list = []
        phone_conf_list = []
         
        response_duration = total_duration
        if len(ctm_info) > 0:
            # response time
            start_time = ctm_info[0][1]
            # start_time + duration
            end_time = ctm_info[-1][1] + ctm_info[-1][2]
            response_duration = end_time - start_time        
        
        for phone, start_time, duration, conf in ctm_info:
            phone_duration_list.append(duration)
            phone_conf_list.append(conf)
            phone_count_dict[phone] += 1  
            
        # strat_time and duration of last phone
        # word in articlulation time
        phone_count = sum(list(phone_count_dict.values()))
        phone_freq = phone_count / response_duration
        phone_duration_stats = get_stats(phone_duration_list, prefix = "phone_duration_")
        phone_conf_stats = get_stats(phone_conf_list, prefix="phone_conf_")
        
        phone_basic_dict = { 
                            "phone_count": phone_count,
                            "phone_freq": phone_freq,
                           }
        
        phone_stats_dict = merge_dict(phone_duration_stats, phone_conf_stats)
        phone_dict = merge_dict(phone_basic_dict, phone_stats_dict)
        
        return phone_dict, response_duration
