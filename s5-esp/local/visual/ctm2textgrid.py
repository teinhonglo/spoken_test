'''
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 0.66 
tiers? <exists> 
size = 1
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 0.66 
        intervals: size = 3 
        intervals [1]:
            xmin = 0 
            xmax = 0.06 
            text = "SIL" 
        intervals [2]:
            xmin = 0.06 
            xmax = 0.24 
            text = "l" 
        intervals [3]:
            xmin = 0.24 
            xmax = 0.66 
            text = "uo4" 
'''
import argparse
import os
from statistics import mean, median

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--corr_phn_fn', default="data/train_tr/gigaspeech/ctm", type=str)
parser.add_argument('--dest_dir', default="data/train_tr/gigaspeech/textgrid", type=str)

args = parser.parse_args()

# correct phoneme
corr_phn_fn = args.corr_phn_fn
# perceived phoneme
dest_dir = args.dest_dir

#correct_phones_dict = {"Phones":[],"Time":[]}
corr_phn_dict = {}
#perceived_phones_dict = {"Phones":[],"Time":[]}
perc_phn_dict = {}
conf_list = []

def file2info(phn_dict, phn_fn, conf_list):
    last_ed = -1
    with open(phn_fn, "r") as fn:
        for line in fn.readlines():
            # ctm format
            # uttid channel start_time duration phone_token confidence
            fname, _, st, dur, phn, conf = line.split("\n")[0].split()
            fname = fname.split(".")[0]
            st = float(st)
            ed = st + float(dur)
            if fname in phn_dict:
                phn_dict[fname]["Phones"].append([phn, str(st), str(ed), conf])
            else:
                if last_ed != -1:
                    phn_dict[last_fname]["Time"] = last_ed
                phn_dict[fname]= {"Phones": [[phn, str(st), str(ed), conf]], "Time":None}
            conf_list.append(conf)
            last_fname = fname
            last_ed = ed
    phn_dict[last_fname]["Time"] = last_ed
    return phn_dict

corr_phn_dict = file2info(corr_phn_dict, corr_phn_fn, [])

def write_interval(tg_fn, phn_list, xmax, name, num_items):
    tg_fn.write("\titem ["+ num_items +"]:\n")
    tg_fn.write("\t\tclass = \"IntervalTier\"\n")
    tg_fn.write("\t\tname = \"" + name + "\"\n")
    tg_fn.write("\t\txmin = 0\n")
    tg_fn.write("\t\txmax = " + str(xmax) + "\n")
    tg_fn.write("\t\tintervals: size = " + str(len(phn_list)) + "\n")
    for i in range(len(list(phn_list))):
        tg_fn.write("\t\tintervals [" + str(i+1) + "]:\n")
        tg_fn.write("\t\t\txmin = " + phn_list[i][1] + "\n")
        tg_fn.write("\t\t\txmax = " + phn_list[i][2] + "\n")
        tg_fn.write("\t\t\ttext = \"" + phn_list[i][0] + "," + phn_list[i][-1] + "\"\n")

if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)

for fname in list(corr_phn_dict.keys()):
    with open(dest_dir + "/" + fname + ".textgrid", "w") as tg_fn:
        tg_fn.write("File type = \"ooTextFile\"\n")
        tg_fn.write("Object class = \"TextGrid\"\n")
        tg_fn.write("\n")
        tg_fn.write("xmin = 0\n")
        corr_xmax = corr_phn_dict[fname]["Time"]
        tg_fn.write("xmax = " + str(corr_xmax) + "\n")
        tg_fn.write("tiers? <exists>\n")
        tg_fn.write("size = 1\n")
        tg_fn.write("item []:\n")
        write_interval(tg_fn, corr_phn_dict[fname]["Phones"], corr_xmax, "correct_phone", "1")
