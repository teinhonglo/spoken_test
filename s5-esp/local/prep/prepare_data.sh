#!/usr/bin/env bash

stage=0
data_name=spoken_test_2022_jan28
model_name=gigaspeech
model_tag="Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave"
replace_text=false
data_root=data

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le -3 ]; then
    find $data_root/$data_name -name "*.wav" -size -45k
    exit 0;
fi

if [ $stage -le -2 ]; then
    ./local/prep/create_decode_data.sh --data_root $data_root --test_sets "$data_name"
fi

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"

if [ $stage -le -1 ]; then
    python local/prep/repaire_and_resample.py --data_dir $data_root/$data_name
fi

conda activate

if [ $stage -le 0 ]; then
    ./local/e2e_stt/extract_feats.sh --data_root $data_root --data_sets $data_name --model_name $model_name --model_tag "$model_tag"
    dest_dir=$data_root/$data_name/$model_name
    
    if [ $replace_text == "true" ]; then
        cp $dest_dir/text $data_root/$data_name/text
    fi
fi

if [ $stage -le 1 ]; then
    dest_dir=$data_root/$data_name/$model_name
    ./local/prep/prepare_xlsx.sh --data_root $data_root --data_name $data_name --dest_dir $dest_dir
fi
