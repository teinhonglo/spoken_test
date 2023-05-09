#!/usr/bin/env bash

stage=0
stop_stage=10000
feats_stage=0
feats_stop_stage=10000
xlsx_stage=0
xlsx_stop_stage=10000
skip_resample="true"
data_name=gept_b1
model_dir=../models/multi_en-cnn_tdnn_1a_train_cleaned_mct
model_name=multi_en_mct_cnn_tdnnf_tgt3meg-dl
graph_affix=_tgt3meg-dl
replace_text=false
data_root=data
max_nj=20
corpus_path=

. ./cmd.sh
. ./path.sh
. parse_options.sh

set -euo pipefail

if [ $stage -le -3 ] && [ $stop_stage -ge -3 ]; then
    find $data_root/$data_name -name "*.wav" -size -45k
    exit 0;
fi

if [ $stage -le -2 ] && [ $stop_stage -ge -2 ]; then
    
    if [ -z $corpus_path ]; then
        corpus_path="$data_root/$data_name/wavs"
    fi
    
    ./local/prep/create_decode_data.sh --data_root $data_root --test_sets "$data_name" --corpus_path $corpus_path
fi

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
    if [ "$skip_resample" != "true" ]; then
        python local/prep/repair_and_resample.py --data_dir $data_root/$data_name
    fi
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    ./local/kaldi_stt/extract_feats.sh  --stage $feats_stage --test_sets $data_name --max-nj $max_nj \
                                        --stop_stage $feats_stop_stage \
                                        --data_root $data_root \
                                        --model_name $model_name --model_dir $model_dir \
                                        --graph_affix $graph_affix
    
    dest_dir=$data_root/$data_name/$model_name
    
    if [ $replace_text == "true" ]; then
        cp $dest_dir/text $data_root/$data_name/text
    fi
fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    dest_dir=$data_root/$data_name/$model_name
    ./local/prep/prepare_xlsx.sh --data_root $data_root --data_name $data_name --dest_dir $dest_dir --stage $xlsx_stage --stop-stage $xlsx_stop_stage
fi



