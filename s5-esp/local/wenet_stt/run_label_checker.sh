#!/usr/bin/env bash

stage=0
data_name=spoken_test_2022_jan28
data_root=data
dir=../models/20210728_u2pp_conformer_server

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 0 ]; then
    label_checker_dir=label_checker
    mkdir -p $label_checker_dir
    
    label_checker_main  --text $data_root/$data_name/text \
                        --wav_scp $data_root/$data_name/wav.scp \
                        --is_penalty 4.6 --del_penalty 2.3 \
                        --sample_rate 16000 --num_threads 8 \
                        --result $label_checker_dir/result_${data_name} \
                        --timestamp $label_checker_dir/timestamp_${data_name} \
                        --model_path $dir/final.zip --dict_path $dir/words.txt
fi
