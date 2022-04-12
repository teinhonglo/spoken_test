import pandas as pd
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="data/voice_2022/wav.scp")

args = parser.parse_args()

data_dir = args.data_dir
wavscp_fn = data_dir + "/wav.scp"
wavscp_dict = {}

# phone, first_name, last_name, degree, grade, gender, email, recording_equipment, 1-1, 2-1, 2-2, 2-3, 2-4, 2-5, 2-6, 2-7, 2-8, 2-9, 2-10, 3-1
titles = ["phone", "first_name", "last_name", "degree", "grade", "gender", "email", "recording_equipment", "date", "1-1", "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9", "2-10", "3-1"]
questions = ["1-1", "2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9", "2-10", "3-1"]

csv_info = {t:[] for t in titles}

def proc_spkinfo(readme_fn):
    spk_info = {t:"0" for t in ["first_name", "last_name", "degree", "grade", "gender", "email", "recording_equipment"]}
    
    with open(readme_fn, "r") as fn:
        for line in fn.readlines():
            info = line.split()[0].split(",")
            if info[0] in spk_info:
                spk_info[info[0]] = info[1]
    
    return spk_info

with open(wavscp_fn, "r") as fn:
    for line in fn.readlines():
        uttid, wav_path = line.split()
        wavscp_dict[uttid] = wav_path

for uttid, wav_path in wavscp_dict.items():
    dir_name = os.path.dirname(wav_path)
    w_dir_name = "/".join(dir_name.split("/")[2:])
    phoneNo = dir_name.split("/")[3]
    csv_info["phone"].append(phoneNo)
    num_egs = len(csv_info["phone"])
    # readme.txt
    readme_fn = os.path.join(dir_name, "readme.txt")
    if os.path.isfile(readme_fn):
        spk_info = proc_spkinfo(readme_fn)
        for k, v in spk_info.items():
            csv_info[k].append(v)
    else:
        for i in ["first_name", "last_name", "degree", "grade", "gender", "email", "recording_equipment"]:
            csv_info[i].append("0")
    
    # other question ID
    for fname in os.listdir(dir_name):    
        fn_ext = fname.split(".")[-1]
        if fn_ext == "wav":
            q_ID = "-".join(fname.split("-")[1:3])
            date = fname.split("-")[-1].split(".")[0]
            csv_info[q_ID].append(os.path.join(w_dir_name, fname))
            if len(csv_info["date"]) < num_egs:
                csv_info["date"].append(date)
    
    # complement question ID
    for q_ID in questions:
        if len(csv_info[q_ID]) < num_egs:
            csv_info[q_ID].append("0")

df = pd.DataFrame(csv_info)
df.to_excel(data_dir + "/info.xlsx", index=False)
