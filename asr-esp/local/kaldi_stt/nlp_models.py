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
        
        self.nlp_tokenize = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse', use_gpu='False', tokenize_pretokenized=tokenize_pretokenized)
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
        vp_list = []
        pos_list = []
        mor_list = []
        dep_list = []
        
        doc = self.nlp_tokenize(text.lower())
        hit_dict = defaultdict(dict)
        
        for si, sent in enumerate(doc.sentences):
            for wi, word in enumerate(sent.words):
                lemma_word = word.lemma
                upos = word.pos
                mor = word.feats
                dep = word.deprel
                
                pos_info[upos] += 1
                
                #if lemma_word in hit_dict and upos in hit_dict[lemma_word]:
                #    continue
                
                try:
                    word_cefr = self.cefr_dict[lemma_word][upos]
                    hit_dict[lemma_word][upos] = 1
                except:
                    word_cefr = None
                
                if word_cefr is not None:
                    vocab_profile[word_cefr] += 1
                
                vp_list.append(word_cefr)
                pos_list.append(upos)
                mor_list.append(mor)
                dep_list.append(dep)
                        
        prefix = "vp_"
        # add prefix
        vocab_feats = {"vp_list": vp_list}
         
        for cefr, num in vocab_profile.items():
            vocab_feats[prefix + cefr] = num
        
        prefix = "pos_"
        pos_feats = {"pos_list": pos_list}
        
        for pos, num in pos_info.items():
            pos_feats[prefix + pos] = num
        
        nlp_feats = merge_dict(vocab_feats, pos_feats)
        
        mor_feats = {"mor_list": mor_list}
        nlp_feats = merge_dict(nlp_feats, mor_feats)
        dep_feats = {"dep_list": dep_list}
        nlp_feats = merge_dict(nlp_feats, dep_feats)
        
        return nlp_feats
       
  
if __name__ == '__main__':
    text = "I agree with the statement. We had many holidays, which is longer than others. Most of us were planning to find a part-time job in order to kill the boring time while we got the news. So it was the same with me.  My friends had work experience and had a more independent life. They are willing to share their interesting work experience with me. In their work life, they met many difficulties. They would try to do something only they are. Different kinds of setbacks made them more strong and going easy on everything afterwards.  I feel the sound of a shampoo."

    nlp_model = NlpModel()
    vp_feats = nlp_model.vocab_profile_feats(text)
    print(vp_feats)
    print(len(vp_feats["pos_list"]))
    print(len(text.split(" ")))



