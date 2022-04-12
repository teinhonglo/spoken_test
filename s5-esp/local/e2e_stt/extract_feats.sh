#!/usr/bin/env bash

set -euo pipefail

data_root=data
data_sets="l2_arctic"
model_name="gigaspeech"
model_tag="Shinji Watanabe/gigaspeech_asr_train_asr_raw_en_bpe5000_valid.acc.ave"
stage=0

. utils/parse_options.sh


. ./path.sh
. ./cmd.sh

if [ $stage -le 0 ]; then
    for data_set in $data_sets; do
        python local/e2e_stt/prepare_feats.py --data_dir $data_root/$data_set --model_name $model_name --model_tag "$model_tag"
    done
fi
