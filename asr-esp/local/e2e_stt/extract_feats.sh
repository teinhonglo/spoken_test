#!/usr/bin/env bash

set -euo pipefail

data_root=data
data_sets="l2_arctic"
model_name="gigaspeech"
model_tag="Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave"
use_streaming=false
stage=0
# vad parameters
vad_mode=0
max_segment_length=15

. ./path.sh
. ./cmd.sh

echo "$0 $@"
. parse_options.sh


set -euo pipefail

if [ $stage -le 0 ]; then
    for data_set in $data_sets; do
        if [ "$use_streaming" == "false" ]; then
            python local/e2e_stt/prepare_feats.py --data_dir $data_root/$data_set --model_name $model_name \
                                                  --model_tag "$model_tag" --vad_mode $vad_mode --max_segment_length $max_segment_length
        else
            python local/e2e_stt/prepare_feats_streaming.py --data_dir $data_root/$data_set --model_name $model_name \
                                              --model_tag "$model_tag" --vad_mode $vad_mode --max_segment_length $max_segment_length
        fi
    done
fi

if [ $stage -le 1 ]; then
    for data_set in $data_sets; do
        compute-wer --text --mode=present ark:$data_root/$data_set/text ark:$data_root/$data_set/$model_name/text > $data_root/$data_set/$model_name/wer
    done
fi
