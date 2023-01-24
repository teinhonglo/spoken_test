import pandas as pd
import os
import argparse
import json
import re

"""
[音檔檔名規則]
u：userSN 學生編號
t ：testSN  場次編號
p ：paperSN  題本編號
i ：itemSN     題目編號
例如： u117_t9_p4_i16_1-2_20220922.wav
表示 學生117_場次9_題本9_題目16_題型一第二題_錄製日期.wav
"""

parser = argparse.ArgumentParser()

parser.add_argument('--anno_path', type=str, default="/share/corpus/2023_teemi/annotation/口說預試試題評分資料-題本4.xlsx")
parser.add_argument('--question_types', type=str, default="基礎聽答,情境式提問與問答")
parser.add_argument('--data_dir', type=str, default="data/pretest/2023_teemi")

args = parser.parse_args()

all_question_types = ["基礎聽答", "情境式提問與問答", "主題式口說任務", "摘要報告", "計分說明"]
sub_question_dict = {"基礎聽答": "5", "情境式提問與問答": "3"}
question_dict = { qt: i + 1 for i, qt in enumerate(all_question_types)}

anno_path = args.anno_path
data_dir = args.data_dir
wavscp_path = os.path.join(data_dir, "wav.scp")
question_types = args.question_types.split(",")

wavscp_dict = {}
anno_df_dict = {qt:pd.read_excel(anno_path, sheet_name=qt) for qt in all_question_types}

with open(wavscp_path, "r") as fn:
    for line in fn.readlines():
        wavid, wav_path = line.split()
        wavscp_dict[wavid] = wav_path
        
wav_path_text = "\n".join(list(wavscp_dict.keys()))

for qt in question_types:
    anno_df = anno_df_dict[qt]
    qt_id = question_dict[qt]
    sub_qnum = sub_question_dict[qt]
    anno_df["數量"] = anno_df["題本"]
    
    for i in range(len(anno_df["題本"])):
        q_num = anno_df["題本"][i]
        s_id = anno_df["學生編號"][i]
        re_pt = "u{}_.+_p{}_.+_{}-[0-9]+_.+".format(s_id, q_num, qt_id)
        wav_info = re.findall(re_pt, wav_path_text)
        rel_sub = str(len(wav_info) - int(sub_qnum))
        wav_paths = ", ".join(wav_info)
        anno_df.at[i, "音檔"] = wav_paths
        anno_df.at[i, "數量"] = rel_sub
        
with pd.ExcelWriter(os.path.basename(anno_path)) as writer:    
    for qt in all_question_types:
        anno_df_dict[qt].to_excel(writer, sheet_name=qt, index=False) 
