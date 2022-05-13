import os
import json
import soundfile
from tqdm import tqdm
from kaldi_models import SpeechModel
from audio_models import AudioModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

parser.add_argument("--model_name",
                    default="gigaspeech",
                    type=str)

parser.add_argument("--gop_result_dir",
                    default="model/model_online/decode/gop",
                    type=str)

parser.add_argument("--gop_json_fn",
                    default="gop_result_dir/json/gop.json",
                    type=str)

parser.add_argument("--sample_rate",
                    default=16000,
                    type=int)

args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name
gop_result_dir = args.gop_result_dir
gop_json_fn = args.gop_json_fn
sample_rate = args.sample_rate

output_dir = os.path.join(data_dir, model_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

wavscp_dict = {}
text_dict = {}
utt_list = []
# stt and ctm
all_info = {}
recog_dict = {}

speech_model = SpeechModel(recog_dict, gop_result_dir, gop_json_fn)
audio_model = AudioModel(sample_rate)

with open(data_dir + "/wav.scp", "r") as fn:
    for i, line in enumerate(fn.readlines()):
        info = line.split()
        wavscp_dict[info[0]] = info[1]
        utt_list.append(info[0])

# anno. (punc)
with open(data_dir + "/text.punc", "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:])

# recog. result
with open(output_dir + "/text", "r") as fn:
    for line in fn.readlines():
        info = line.split()
        recog_dict[info[0]] = " ".join(info[1:])


for i, uttid in tqdm(enumerate(utt_list)):
    wav_path = wavscp_dict[uttid]
    text_prompt = text_dict[uttid]
    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    speech, rate = soundfile.read(wav_path)
    assert rate == sample_rate
    total_duration = speech.shape[0] / rate
    # audio feature
    _, f0_info = audio_model.get_f0(speech)
    _, energy_info = audio_model.get_energy(speech)
    # fluency feature and confidence feature
    text = speech_model.recog(uttid)
    # alignment (stt)
    ctm_info = speech_model.get_ctm(uttid)
    phone_ctm_info, phone_text = speech_model.get_phone_ctm(ctm_info)
    
    sil_feats_info, response_duration = speech_model.sil_feats(ctm_info, total_duration)
    word_feats_info, response_duration = speech_model.word_feats(ctm_info, total_duration)
    phone_feats_info, response_duration = speech_model.phone_feats(phone_ctm_info, total_duration)
    
    all_info[uttid] = { "stt": text, "stt(g2p)": phone_text, "prompt": text_prompt,
                        "wav_path": wav_path, "ctm": ctm_info, 
                        "feats": {  **f0_info, **energy_info, 
                                    **sil_feats_info, **word_feats_info,
                                    **phone_feats_info,
                                    "total_duration": total_duration,
                                    "response_duration": response_duration}}

print(output_dir)
with open(output_dir + "/all.json", "w") as fn:
    json.dump(all_info, fn, indent=4, ensure_ascii=False)

# write STT Result to file
with open(output_dir + "/text", "w") as fn:
    for uttid in utt_list:
        fn.write(uttid + " " + all_info[uttid]["stt"] + "\n")

# write alignment results fo file
with open(output_dir + "/ctm", "w") as fn:
    end_time = -100000
    for uttid in utt_list:
        ctm_infos = all_info[uttid]["ctm"]
        for i in range(len(ctm_infos)):
            text_info, start_time, duration, conf = ctm_infos[i]
            # uttid channel start_time duration text conf
            ctm_info = " ".join([uttid, "1", str(start_time), str(duration), text_info, str(conf)])
            fn.write(ctm_info + "\n")
