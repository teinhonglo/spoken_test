import os
import json
import soundfile
from tqdm import tqdm
from espnet_models import SpeechModel
from audio_models import AudioModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

parser.add_argument("--model_tag",
                    default="Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave",
                    type=str)

parser.add_argument("--model_name",
                    default="google_en",
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

speech_model = SpeechModel(tag)
audio_model = AudioModel()

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
    response_duration = speech.shape[0] / rate
    # audio feature
    _, f0_info = audio_model.get_f0(speech)
    _, energy_info = audio_model.get_energy(speech)
    # fluency feature and confidence feature
    text = speech_model.recog(speech)
    # alignment (stt)
    ctm_info = speech_model.get_ctm(speech, text)
    sil_feats_info = speech_model.sil_feats(ctm_info, response_duration)
    word_feats_info = speech_model.word_feats(ctm_info, response_duration)
    all_info[utt_id] = {"stt": text, "wav_path": wav_path, "ctm": ctm_info, **f0_info, **energy_info, **sil_feats_info, **word_feats_info}

print(output_dir)
with open(output_dir + "/all.json", "w") as fn:
    json.dump(all_info, fn, indent=4)

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
