#!/usr/bin/env bash

stage=0
feats_stage=0
data_name=spoken_test_2022_jan28
model_dir=../models/Librispeech-model-mct-tdnnf
model_name=librispeech_mct_tdnnf_kaldi_tgt3
graph_affix=_tgt3
replace_text=false
data_root=data

. ./cmd.sh
. ./path.sh
. parse_options.sh

set -euo pipefail

if [ $stage -le -3 ]; then
    find $data_root/$data_name -name "*.wav" -size -45k
    exit 0;
fi

if [ $stage -le -2 ]; then
    ./local/prep/create_decode_data.sh --data_root $data_root --test_sets "$data_name"
fi

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"

if [ $stage -le -1 ]; then
    python local/prep/repair_and_resample.py --data_dir $data_root/$data_name
fi

conda activate

if [ $stage -le 0 ]; then
    ./local/kaldi_stt/extract_feats.sh  --stage $feats_stage --test_sets $data_name \
                                        --data_root $data_root \
                                        --model_name $model_name --model_dir $model_dir \
                                        --graph_affix $graph_affix
    dest_dir=$data_root/$data_name/$model_name
    
    if [ $replace_text == "true" ]; then
        cp $dest_dir/text $data_root/$data_name/text
    fi
fi

if [ $stage -le 1 ]; then
    dest_dir=$data_root/$data_name/$model_name
    ./local/prep/prepare_xlsx.sh --data_root $data_root --data_name $data_name --dest_dir $dest_dir
fi
