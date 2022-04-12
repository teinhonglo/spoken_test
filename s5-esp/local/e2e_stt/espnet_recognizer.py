from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.asr_align import CTCSegmentation

import os
import json
import soundfile
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

parser.add_argument("--model_tag",
                    default="Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave",
                    type=str)

parser.add_argument("--model_name",
                    default="gigaspeech",
                    type=str)

args = parser.parse_args()


data_dir = args.data_dir
model_name = args.model_name
output_dir = os.path.join(data_dir, model_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tag = args.model_tag
wavscp_dict = {}
text_dict = {}
utt_list = []
# stt and ctm
all_info = {}

# =============================================================================
d=ModelDownloader(cachedir="./downloads")
asr_model=d.download_and_unpack(tag)

speech2text = Speech2Text.from_pretrained(
    **asr_model,
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1
)

# ==============================================================================

aligner = CTCSegmentation(**asr_model, fs=16000, kaldi_style_text=False)

with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

with open(data_dir + "/text", "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:])

for utt_id in tqdm(utt_list):
    wav_path = wavscp_dict[utt_id]
    text_prompt = text_dict[utt_id]
    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    speech, rate = soundfile.read(wav_path)
    nbests = speech2text(speech)
    text_stt, *_ = nbests[0]
    # alignment (stt)
    segments = aligner(speech, text_stt.split())
    timing_info = segments.segments
    text_info = segments.text
    all_info[utt_id] = {"stt": text_stt, "ctm": []}

    for i in range(len(timing_info)):
        start_time, end_time, conf = timing_info[i]
        start_time = round(start_time, 4)
        end_time = round(end_time, 4)
        duration = round(end_time - start_time, 4)
        conf = round(conf, 4)
        all_info[utt_id]["ctm"].append([text_info[i], start_time, duration, conf])


print(output_dir)
# write STT Result to file
with open(output_dir + "/text", "w") as fn:
    for utt_id in utt_list:
        fn.write(utt_id + " " + all_info[utt_id]["stt"] + "\n")

# write alignment results fo file
with open(output_dir + "/ctm", "w") as fn:
    end_time = -100000
    for utt_id in utt_list:
        ctm_infos = all_info[utt_id]["ctm"]
        for i in range(len(ctm_infos)):
            text_info, start_time, duration, conf = ctm_infos[i]
            # utt_id channel start_time duration text conf
            ctm_info = " ".join([utt_id, "1", str(start_time), str(duration), text_info, str(conf)])
            fn.write(ctm_info + "\n")

with open(output_dir + "/all.json", "w") as fn:
    json.dump(all_info, fn, indent=4)

