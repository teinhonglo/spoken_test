import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="data/voice_2022/text")

parser.add_argument('--dest_dir', type=str, default="data/voice_2022/gigaspeech")

args = parser.parse_args()

data_dir = args.data_dir
dest_dir = args.dest_dir
text_fn = os.path.join(data_dir, "text")

# filename, transcript (stt), transcript (human), notes
titles = ["filename", "trans_stt", "trans_human", "notes"]

xlsx_info = {t:[] for t in titles}

with open(text_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        uttid = info[0]
        content = " ".join(info[1:]).lower()
        
        xlsx_info["filename"].append(uttid)
        xlsx_info["trans_stt"].append(content)
        xlsx_info["trans_human"].append("")
        xlsx_info["notes"].append("")
        

df = pd.DataFrame(xlsx_info)
df.to_excel(os.path.join(data_dir, "trans.xlsx"), index=False)
