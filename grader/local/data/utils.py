import csv
import sys
import numpy as np


def read_corpus(path, num_labels, score_name, corpus="teemi"):
    ids, prompts, levels, sents, wav_paths, extra_embs = [], [], [], [], [], []
    #nlp_model = NlpModel()

    lines = _read_tsv(path)
    for i, line in enumerate(lines):
        if i == 0:
            columns = {key:header_index for header_index, key in enumerate(line)}
            continue
        
        wav_path_list = line[columns['wav_path']].split(" | ")
        text_list = line[columns['text']].split(" | ")
        
        #wav_path_list = [ " ".join(wav_path_list) ]
        #text_list = [ " ".join(text_list) ]
       
        for j in range(len(text_list)):
            # remove a leading- or tailing-space of the utterance.
            wav_path = " ".join(wav_path_list[j].split())
            text = " ".join(text_list[j].split()).split()
            #cefr_emb = [nlp_model.vocab_profile_feats(text_list[j])["vp_" + feats_id ] for feats_id in ["a1", "a2", "b1", "b2", "c1", "c2"]]
            
            text_id = line[columns["text_id"]]
            ids.append(text_id)
            if corpus == "teemi":
                item_code, _, _, _, _, item_no, _ = text_id.split("_")
                prompt = (item_code + "_0" + item_no.split("-")[-1]).lower()
            elif corpus == "icnale":
                prompt = text_id.split("_")[2]
            else:
                prompt = ""
                
            prompts.append(prompt)
            
            levels.append(float(line[columns[score_name]]) - 1)  # Convert 1-8 to 0-7
            sents.append(text)
            wav_paths.append(wav_path)
            extra_embs.append(float(line[columns[score_name]]) - 1)

    levels = np.array(levels)

    return levels, {"ids": ids, "prompts": prompts, "sents": sents, "wav_paths": wav_paths, "extra_embs": extra_embs}

def read_corpus_phonics(path, num_labels, score_name, corpus="phonics", ignored_columns=["text_id", "fluency", "accuracy"]):
    ids, feats, feat_keys, levels = [], [], [], []
    #nlp_model = NlpModel()

    lines = _read_tsv(path)
    for i, line in enumerate(lines):
        
        if i == 0:
            columns = {key:header_index for header_index, key in enumerate(line)}
            feat_keys = [ k for k in list(columns.keys()) if k not in ignored_columns ]
            continue
       
        text_id = line[columns["text_id"]]
        ids.append(text_id)
        feats.append([float(line[columns[k]]) for k in list(columns.keys()) if k not in ignored_columns])
        levels.append(float(line[columns[score_name]])) # 0-4 for fluency scores; 0-6 for accuracy scores

    levels = np.array(levels)

    return levels, {"ids": ids, "feats": feats, "feat_keys": feat_keys}


def _read_tsv(input_file, quotechar=None):
    print(input_file)
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines
