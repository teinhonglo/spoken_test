import os
import numpy as np
import json
import soundfile
from collections import defaultdict
from tqdm import tqdm
from g2p_en import G2p
from whisper.normalizers import EnglishTextNormalizer
import whisperx
from whisper.tokenizer import get_tokenizer
import string
import re


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
    def __init__(self, tag="large-v2", device="cuda", language="en", condition_on_previous_text=False):
        # Fluency
        self.sil_seconds = 0.145
        self.long_sil_seconds = 0.495
        self.vowels = [ "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW" ]
        self.disflunecy_words = ["AH", "UM", "UH", "EM", "OH", "HM", "HMM", 
                                 "Ah", "Um", "Uh", "Em", "Oh", "Hm", "Hmm", 
                                 "ah", "um", "uh", "em", "oh", "hm", "hmm"]
        self.special_words = ["<UNK>"]
        self.g2p = G2p()
        # STT
        #encourage model to transcribe words literally
        tokenizer = get_tokenizer(multilingual=False)  # use multilingual=True if using multilingual model
        
        number_tokens = [
            i
            for i in range(tokenizer.eot)
            if all(c in "0123456789" for c in tokenizer.decode([i]).removeprefix(" "))
        ]
        
        punc_tokens = [
            i 
            for i in range(tokenizer.eot)
            if all(c in list("!?-,.。，")  for c in tokenizer.decode([i]).removeprefix(" "))
        ]
        
        suppress_tokens = [-1] + number_tokens #+ punc_tokens
        
        self.compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
        self.batch_size = 4
        self.language = language
        self.decode_options = {"suppress_tokens": suppress_tokens}
        self.decode_options["condition_on_previous_text"] = condition_on_previous_text
        self.decode_options["suppress_numerals"] = True
        self.device = device
        # stt model
        self.model = whisperx.load_model(tag, self.device, compute_type=self.compute_type, language=self.language, asr_options=self.decode_options)
        # alignment model
        self.model_a, self.metadata = whisperx.load_align_model(language_code=self.language, device=self.device)
        # english std
        self.eng_std = EnglishTextNormalizer()

    
    # STT features
    def recog(self, audio):
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        # merged results
        segments = [ result['segments'][i]['text'] for i in range(len(result['segments'])) ]
        timestamp = [ [result['segments'][i]['start'], result['segments'][i]['end']] for i in range(len(result['segments'])) ]
        text = " ".join(segments)
        text_norm = re.sub(r'[^\w\s]', '', text)
        results = {"segments": [{"text": text_norm, "start": timestamp[0][0], "end": timestamp[-1][-1]}], "language": self.language }
        
        result = whisperx.align(results["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False)
        '''
        [{'start': 0.929, 'end': 4.753, 'text': ' I think Chinese class is the most interesting.', 'words': [{'word': 'I', 'start': 0.929, 'end': 1.029, 'score': 0.996}, {'word': 'think', 'start': 1.109, 'end': 1.389, 'score': 0.816}, {'word': 'Chinese', 'start': 1.489, 'end': 2.01, 'score': 0.677}, {'word': 'class', 'start': 2.11, 'end': 2.551, 'score': 0.924}, {'word': 'is', 'start': 2.691, 'end': 2.811, 'score': 0.816}, {'word': 'the', 'start': 2.891, 'end': 3.031, 'score': 0.771}, {'word': 'most', 'start': 3.111, 'end': 3.632, 'score': 0.816}, {'word': 'interesting.', 'start': 4.032, 'end': 4.753, 'score': 0.569}]}]
        '''
        word_ctm = []

        #print("=======")
        #print("Origin:", text)
        #print("Text_Norm:", text_norm)
        for word_info in result["segments"][0]['words']:
            #print(word_info)
            if 'start' not in word_info:
                continue
            # [ word, start, duration, score ]
            word_ctm.append([word_info['word'], word_info['start'], word_info['end'] - word_info['start'], word_info['score']])
        
        phone_ctm, _ = self.get_phone_ctm(word_ctm)
        
        return [text, text_norm], [word_ctm, phone_ctm]
    
    
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
        articulation rate
        '''
        word_count_dict = defaultdict(int)
        word_duration_list = []
        word_conf_list = []
        word_charlen_list = []
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
            word_charlen_list.append(len(word))
            word_count_dict[word] += 1
            
            if word in self.disflunecy_words:
                num_disfluecy += 1
            
            if word in prev_words:
                num_repeat += 1
            
            prev_words = [word]
            
        # strat_time and duration of last word
        # word in articlulation time
        word_duration_stats = get_stats(word_duration_list, prefix = "word_duration_")
        word_conf_stats = get_stats(word_conf_list, prefix="word_conf_")
        word_charlen_stats = get_stats(word_charlen_list, prefix="word_charlen_")
        
        # word basic_dict
        word_count = sum(list(word_count_dict.values()))
        word_distinct = len(list(word_count_dict.keys()))
        word_freq = word_count / response_duration
        
        word_basic_dict = { 
                            "word_count": word_count,
                            "word_distinct": word_distinct,
                            "word_freq": word_freq,
                            "word_num_disfluency": num_disfluecy,
                            "word_num_repeat": num_repeat
                          }
        
        word_stats_dict = merge_dict(word_duration_stats, word_conf_stats)
        word_stats_dict = merge_dict(word_charlen_stats, word_stats_dict)
        word_dict = merge_dict(word_basic_dict, word_stats_dict)
        
        return word_dict, response_duration
    
     
    def phone_feats(self, ctm_info, total_duration):
        phone_count_dict = defaultdict(int)
        phone_duration_list = []
        phone_conf_list = []
        vowel_duration_list = []
        vowel_conf_list = []
         
        response_duration = total_duration
        
        if len(ctm_info) > 0:
            # response time
            start_time = ctm_info[0][1]
            # start_time + duration
            end_time = ctm_info[-1][1] + ctm_info[-1][2]
            response_duration = end_time - start_time        
        
        for phone, start_time, duration, conf in ctm_info:
            phone = phone.rstrip(string.digits+'*_')
            phone_duration_list.append(duration)
            phone_conf_list.append(conf)
            phone_count_dict[phone] += 1
            
            if phone in self.vowels:
                vowel_duration_list.append(duration)
                vowel_conf_list.append(conf)
            
        # strat_time and duration of last phone
        # word in articlulation time
        phone_count = sum(list(phone_count_dict.values()))
        phone_freq = phone_count / response_duration
        phone_duration_stats = get_stats(phone_duration_list, prefix = "phone_duration_")
        phone_conf_stats = get_stats(phone_conf_list, prefix="phone_conf_")
        vowel_duration_stats = get_stats(vowel_duration_list, prefix = "vowel_duration_")
        vowel_conf_stats = get_stats(vowel_conf_list, prefix="vowel_conf_")
        
        phone_basic_dict = { 
                            "phone_count": phone_count,
                            "phone_freq": phone_freq,
                           }
        
        phone_stats_dict = merge_dict(phone_duration_stats, phone_conf_stats)
        vowel_stats_dict = merge_dict(vowel_duration_stats, vowel_conf_stats)
        phone_stats_dict = merge_dict(phone_stats_dict, vowel_stats_dict)
        phone_dict = merge_dict(phone_basic_dict, phone_stats_dict)
        
        return phone_dict, response_duration
