#!/usr/bin/env bash

set -euo pipefail

data_root=data
data_sets="l2_arctic"
model_name="large"
model_tag="large"
use_streaming=false
stage=0
# vad parameters
vad_mode=0
max_segment_length=15
lang="english"

. ./path.sh
. ./cmd.sh

echo "$0 $@"
. utils/parse_options.sh


set -euo pipefail

if [ $stage -le 0 ]; then
    for data_set in $data_sets; do
        #if [ -f $data_root/$data_set/$model_name/text ]; then
        #    continue
        #fi
        
        if [ "$use_streaming" == "false" ]; then
            python local/e2e_stt/prepare_feats_whisper.py --data_dir $data_root/$data_set --model_name $model_name \
                                                  --model_tag "$model_tag" --vad_mode $vad_mode --max_segment_length $max_segment_length --lang $lang
        else
            python local/e2e_stt/prepare_feats_whisper.py --data_dir $data_root/$data_set --model_name $model_name \
                                              --model_tag "$model_tag" --vad_mode $vad_mode --max_segment_length $max_segment_length --lang $lang
        fi
    done
fi

if [ $stage -le 1 ]; then
    for data_set in $data_sets; do
        for text in text.1p text.2p text.3p text; do
            ref=$data_root/$data_set/$text
            hyp=$data_root/$data_set/$model_name/text
            
            if [ ! -f $ref ]; then
                echo "No such file: $ref";
                continue;
            fi
            
            echo "$model_tag $text"
            python local/e2e_stt/calc_wer.py --ref $ref --hyp $hyp
        done
    done
fi