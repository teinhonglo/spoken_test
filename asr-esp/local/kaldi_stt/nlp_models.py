import os
import numpy as np
import json
import soundfile
from collections import defaultdict
from tqdm import tqdm
from g2p_en import G2p
import re
import stanza
import pandas as pd


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
    
    
class NlpModel(object):
    def __init__(self, tokenize_pretokenized=False, 
                cefr_dict_path="/share/nas167/teinhonglo/AcousticModel/spoken_test/corpus/speaking/CEFR-J_Wordlist_Ver1.6.xlsx"):
                #cefr_dict_path="/share/nas167/teinhonglo/AcousticModel/spoken_test/corpus/speaking/CEFR-J_Wordlist_Ver1.6_with_C1C2.xlsx"):
        
        self.nlp_tokenize = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', use_gpu='False', tokenize_pretokenized=tokenize_pretokenized)
        self.cefr_dict = self.__build_cefr_dict(cefr_dict_path)
        self.cefr_levels = ["a1", "a2", "b1", "b2"]
        self.pos_tags = ["ADJ", "ADP", "ADV", "AUX", 
                        "CCONJ", "DET", "INTJ", "NOUN", 
                        "NUM", "PART", "PRON", "PROPN", 
                        "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    
    def __build_cefr_dict(self, cefr_dict_path):
        cefr_dict = defaultdict(dict)
        pos_conv_dict = {}
        
        cefr_vocab_df = pd.read_excel(cefr_dict_path, sheet_name="ALL", usecols=["headword", "pos", "CEFR"])
        pos_conv_df = pd.read_excel(cefr_dict_path, sheet_name="POS", usecols=["upos", "pos"])
        
        for i in range(len(pos_conv_df["pos"])):
            pos_conv_dict[pos_conv_df["pos"][i]] = pos_conv_df["upos"][i]
        
        for i in range(len(cefr_vocab_df["headword"])):
            words = cefr_vocab_df["headword"][i]
            pos = cefr_vocab_df["pos"][i]
            cefr = cefr_vocab_df["CEFR"][i]
            
            for upos in pos_conv_dict[pos].split("/"):
                for word in words.split("/"):
                    cefr_dict[word][upos] = cefr.lower()
            
        return cefr_dict
     
    def vocab_profile_feats(self, text):
        vocab_profile = {cl:0 for cl in self.cefr_levels}
        pos_info = {pt: 0 for pt in self.pos_tags}
        
        doc = self.nlp_tokenize(text.lower())
        hit_dict = defaultdict(dict)
        
        for si, sent in enumerate(doc.sentences):
            for wi, word in enumerate(sent.words):
                lemma_word = word.lemma
                upos = word.pos
                pos_info[upos] += 1
                
                if lemma_word in hit_dict and upos in hit_dict[lemma_word]:
                    continue
                
                try:
                    word_cefr = self.cefr_dict[lemma_word][upos]
                    hit_dict[lemma_word][upos] = 1
                except:
                    word_cefr = None
                
                if word_cefr is not None:
                    vocab_profile[word_cefr] += 1
        
        prefix = "vp_"
        # add prefix
        vocab_feats = {}
         
        for cefr, num in vocab_profile.items():
            vocab_feats[prefix + cefr] = num
        
        prefix = "pos_"
        pos_feats = {}
        for pos, num in pos_info.items():
            pos_feats[prefix + pos] = num
        
        nlp_feats = merge_dict(vocab_feats, pos_feats)
        
        return nlp_feats
       
  
if __name__ == '__main__':
    text = "I'M TAKING A TEST AND I'M BORED HIM | WELL MY FAVORITE MUSIC IS COUNTRY MUSIC BECAUSE IT HELPS ME RELAX | I WAS THIRTEEN YEARS OLD WHEN I WENT TO JUNIOR HIGH SCHOOL GYM CLOTHES | WE SHOULDN'T TALK LOUDLY ON THE TRAIN IN RESTAURANTS FOR EXAMPLE | I WOULD KEEP TAKING A SHOWER WHEN THE LIGHTS ARE OUT | I WOULD FEEL MORE RELAXED IN THE EVENING BECAUSE I USUALLY EXERCISE AFTERWARD TAKE A SHOWER AND THEN WHERE SOME MORE COMFORT COMFORTABLE CLOTHES SO AND WATCH MY VERY SERIOUS AND THESE THINGS MAKE ME RELAXED | BECAUSE FAST FOOD IN FAST FOOD THERE IS USUALLY ISN'T ENOUGH FIBRE AND THEIR USE OF FIFA FUCK FAST FOOD IS USUALLY FRIED AND IS OFTEN EASIER THAN OTHER FOOD AND I DON'T KNOW NUTRITIOUS ENOUGH | I THINK I CAN MORE EASILY ANGRY AT WORK AND I DON'T KNOW JOGGING CALMS ME DOWN AND I LIKE TO BE ALONE WHEN I GET ANGRY I DON'T WANT TO BE IN A CROWDED PLACES WHEN I'M FEELING UPSET | WELL I WENT HOME ON LABOR DAY WEEKEND AND I PAID WITH MY LITTLE NIECE SO AH I I GAVE MY LITTLE NIECE A SMALL PATCH OF DOUGH IN TARTAR HOW TOO NEAT AND SHAPE THE DOUGH AND I ALSO PUT SOME FEELING I HAVE MOSTLY MASH SWEET POTATOES IN JUDAH DOUGH AND | I RATHER SEE A WESTERN MOVIE BECAUSE WESTERN MOVIES ARE USUALLY MORE CREATIVE IN THEIR CLASS AND THEY HAVE USUALLY WITH YOUR LINES AND LIKE I GUESS THESE TWO WESTERN NUMERACY AND I LIKE MOVIES"
    nlp_model = NlpModel()
    vp_feats = nlp_model.vocab_profile_feats(text)
    print(vp_feats)



