import re
import json
import os

import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default = "data/icnale/icnale_monologue/whisperx_large-v1",
                    type=str)

args = parser.parse_args()

data_dir = args.data_dir
json_fn = data_dir + "/all.json"
feats_json_fn = data_dir + "/aspect_feats.json"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

POS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", 
       "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", 
       "PUNCT", "SCONJ", "SYM", "VERB", "X"]

MOR = ['VerbForm=Ger', 'VerbForm=Conv', 'Mood=Prp', 'Animacy=Hum', 'Case=Gen', 'Aspect=Prog', 
       'NumType=Ord', 'Number=Count', 'Case=Loc', 'Number=Pauc', 'Case=Abe', 'Case=Ill', 
       'Case=Voc', 'Case=Sup', 'Case=Par', 'NounClass=Bantu20', 'Case=Ine', 'NounClass=Bantu15', 
       'Voice=Lfoc', 'Degree=Dim', 'Mood=Cnd', 'NounClass=Bantu10', 'Case=Erg', 'Voice=Rcp', 
       'Number=Inv', 'PronType=Emp', 'Mood=Des', 'Tense=Past', 'Degree=Abs', 'Polite=Form', 
       'Clusivity=In', 'NounClass=Bantu3', 'Animacy=Nhum', 'Case=Tem', 'NounClass=Wol4', 'Person=1', 
       'Evident=Nfh', 'Mood=Opt', 'NounClass=Bantu13', 'NounClass=Wol10', 'Voice=Act', 'NounClass=Bantu4', 
       'Polarity=Pos', 'Polite=Humb', 'NumType=Frac', 'Gender=Com', 'Case=Equ', 'Case=Per', 
       'Case=Ela', 'Voice=Bfoc', 'Degree=Aug', 'Number=Coll', 'Number=Tri', 'Typo=Yes', 
       'Aspect=Iter', 'Case=Ins', 'Voice=Cau', 'NumType=Sets', 'PronType=Neg', 'Voice=Antip', 
       'Case=Cns', 'Degree=Pos', 'Aspect=Imp', 'Definite=Def', 'VerbForm=Fin', 'Case=Cmp', 
       'Voice=Pass', 'Mood=Pot', 'Case=Spl', 'Person=0', 'Person=4', 'Definite=Spec', 
       'Case=Add', 'NounClass=Bantu19', 'NumType=Dist', 'Number=Dual', 'PronType=Ind', 'Number=Sing', 
       'NounClass=Bantu6', 'Number=Grpa', 'VerbForm=Inf', 'Animacy=Anim', 'PronType=Prs', 'Case=Com', 
       'NounClass=Bantu5', 'PronType=Tot', 'NounClass=Wol8', 'Polite=Elev', 'NounClass=Wol2', 'NounClass=Bantu14', 
       'Case=Acc', 'Case=Sub', 'NounClass=Bantu16', 'NounClass=Wol7', 'VerbForm=Part', 'NumType=Range', 
       'Voice=Inv', 'NounClass=Bantu9', 'Polite=Infm', 'NounClass=Wol9', 'VerbForm=Gdv', 'Tense=Pres', 
       'Abbr=Yes', 'NumType=Mult', 'Definite=Com', 'Case=Ade', 'NounClass=Wol12', 'NounClass=Bantu18', 
       'Case=Dat', 'NounClass=Bantu23', 'Case=Sbl', 'Case=Ter', 'Gender=Fem', 'Case=Abl', 
       'Mood=Jus', 'Mood=Imp', 'NounClass=Bantu7', 'PronType=Int', 'NounClass=Wol3', 'NumType=Card', 
       'NounClass=Bantu17', 'Aspect=Perf', 'Mood=Ind', 'PronType=Rcp', 'Aspect=Hab', 'Degree=Cmp', 
       'Evident=Fh', 'Case=Nom', 'Tense=Fut', 'Case=Dis', 'Tense=Imp', 'Case=Ess', 
       'Mood=Int', 'Gender=Neut', 'NounClass=Bantu12', 'VerbForm=Vnoun', 'Gender=Masc', 'Case=All', 
       'Tense=Pqp', 'Mood=Qot', 'Number=Ptan', 'Voice=Mid', 'NounClass=Bantu2', 'Case=Sbe', 
       'Mood=Nec', 'Mood=Sub', 'Number=Plur', 'Foreign=Yes', 'Degree=Equ', 'Reflex=Yes', 
       'Number=Grpl', 'Voice=Dir', 'Aspect=Prosp', 'NounClass=Wol5', 'Person=3', 'Case=Cau', 
       'VerbForm=Sup', 'Poss=Yes', 'NounClass=Bantu8', 'PronType=Art', 'Case=Tra', 'Case=Abs', 
       'PronType=Rel', 'Mood=Adm', 'PronType=Exc', 'NounClass=Bantu22', 'PronType=Dem', 'Definite=Ind', 
       'NounClass=Wol6', 'NounClass=Bantu1', 'Person=2', 'Case=Lat', 'Case=Ben', 'Polarity=Neg', 
       'Clusivity=Ex', 'Definite=Cons', 'Case=Core:', 'Case=Del', 'Animacy=Inan', 'Mood=Irr', 'Degree=Sup', # sharon's data
       'ExtPos=PRON', 'NumForm=Word', 'NumForm=Digit', 'ExtPos=ADP', 'ExtPos=ADV', # ICNALE
       'Style=Vrnc', 'NumForm=Roman']

DEP = ['root', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 
       'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 
       'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 
       'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 
       'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 
       'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp', 
       'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 
       'punct', 'quantmod', 'relcl', 'xcomp', # sharon's data
       'obl', 'cop', 'discourse', 'orphan', 'flat', 'dislocated', 
       'vocative', 'obj', 'iobj', 'reparandum', 'fixed', # ICNALE
       'goeswith', 'list']

print("POS:", len(POS), "MOR:", len(MOR), "DEP:", len(DEP))

pos_map_dict = {p: i for i, p in enumerate(POS)}
mor_map_dict = {m: i for i, m in enumerate(MOR)}
dep_map_dict = {d: i for i, d in enumerate(DEP)}

extra_pos = set()
extra_mor = set()
extra_dep = set()

with open(json_fn, "r") as fn:
    all_info = json.load(fn)

def get_stats(word_info, raw_list):
    start_time, end_time = word_info
    st_idx = int(start_time * 100)
    ed_idx = int(end_time * 100)
    
    stats_np = np.array(raw_list[st_idx: ed_idx])
    stats_np = stats_np[np.nonzero(stats_np)]

    number = len(stats_np)

    if number == 0:
        summ, mean, std, median, mad, maximum, minimum = 0, 0, 0, 0, 0, 0, 0
    else:
        summ = np.sum(stats_np)
        mean = np.mean(stats_np)
        std = np.std(stats_np)
        median = np.median(stats_np)
        mad = np.sum(np.absolute(stats_np - mean)) / number
        maximum = np.max(stats_np)
        minimum = np.min(stats_np)
    
    return [mean, std, median, mad, maximum, minimum]

multi_aspect_feats = {}

for uttid in tqdm(list(all_info.keys())):
    multi_aspect_feats[uttid] = {"delivery": [], "language_use": []}
    text = all_info[uttid]["stt"]
    text_norm = re.sub(r'[^\w\s]', '', text)
    
    '''
    Word-level delivery features
    The features include 
        duration(1), pitch(7), intensity(7),
        following silence or pause length(1), 
        [posterior probabilities of AMs from both native ASR and non-native ASR, 
        LM score from non-native ASR,] and confidence score of non-native ASR 
    '''
    delivery_feats = []

    num_words = len(all_info[uttid]["word_ctm"])
    # f0
    f0_list = all_info[uttid]["feats"]["f0_list"]
    # energy
    energy_list = all_info[uttid]["feats"]["energy_rms_list"]

    #print("word-level length", len(all_info[uttid]["word_ctm"]))
    for i in range(len(all_info[uttid]["word_ctm"])):
        word, start_time, duration, confidence = all_info[uttid]["word_ctm"][i]
        start_time = float(start_time)
        duration = float(duration)
        confidence = float(confidence)
        end_time = start_time + duration

        word_info = [start_time, end_time]
        # f0
        f0_feats = get_stats(word_info, f0_list)
        # energy
        energy_feats = get_stats(word_info, energy_list)
        
        if i < len(all_info[uttid]["word_ctm"]) - 1:
            following_word, following_start_time, following_duration, following_confidence = all_info[uttid]["word_ctm"][i+1]
            following_start_time = float(following_start_time)
            following_duration = float(following_duration)
            following_pause_length = following_start_time - end_time
        else:
            following_pause_length = 0.0

        delivery_feats.append([duration] + f0_feats + energy_feats + [following_pause_length, confidence])

    multi_aspect_feats[uttid]["delivery"] = delivery_feats
    '''
    Token-level language use features
    '''
    pos_list = all_info[uttid]["feats"]["pos_list"]
    mor_list = all_info[uttid]["feats"]["mor_list"]
    dep_list = all_info[uttid]["feats"]["dep_list"]
   
    assert len(pos_list) == len(mor_list)
    assert len(pos_list) == len(dep_list)
    
    lang_use_feats = []
    #print("token-level length", len(pos_list))
    for i in range(len(pos_list)):
        pos_feats = np.zeros(len(pos_map_dict))
        mor_feats = np.zeros(len(mor_map_dict))
        dep_feats = np.zeros(len(dep_map_dict))
        
        pos = pos_list[i]
        mor = mor_list[i]
        dep = dep_list[i]

        # pos
        if pos in pos_map_dict:
            pos_feats[pos_map_dict[pos]] = 1
        else:
            extra_pos.add(pos)
        # mor
        if mor is not None:
            for m in mor.split("|"):
                if m in mor_map_dict:
                    mor_feats[mor_map_dict[m]] = 1
                else:
                    extra_mor.add(m)
        # dep
        if ":" in dep:
            dep = dep.split(":")[0]

        if dep in dep_map_dict:
            dep_feats[dep_map_dict[dep]] = 1
        else:
            extra_dep.add(dep)
        
        lang_use_feats.append(np.concatenate((pos_feats, mor_feats, dep_feats)).tolist())
    
    multi_aspect_feats[uttid]["language_use"] = lang_use_feats

with open(feats_json_fn, "w") as fn:
    json.dump(multi_aspect_feats, fn, indent=4, ensure_ascii=False, cls=NpEncoder)

print("pos", extra_pos)
print("mor", extra_mor)
print("dep", extra_dep)
