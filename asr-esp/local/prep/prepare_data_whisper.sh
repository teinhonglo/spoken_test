#!/usr/bin/env bash

stage=0
stop_stage=100000
feats_stage=0
feats_stop_stage=100000
data_root=data/icnale
data_name=icnale_monologue
skip_resample="true"
replace_text=false
use_streaming=false
use_prep_v2=false
# whisper parameters
model_name=whisperx_large-v1
model_tag="large-v1"
use_whisperx="true"
extra_options="--suppress_numeric_tokens --suppress_punc_tokens"
lang="en" # en for whisperx
corpus_path=

echo "$0 $@"
. utils/parse_options.sh

. ./cmd.sh
. ./path.sh

set -euo pipefail

if [ ${stage} -le -3 ] && [ ${stop_stage} -ge -3 ]; then
    find $data_root/$data_name -name "*.wav" -size -45k
    exit 0;
fi

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
    
    if [ -z $corpus_path ]; then
        corpus_path="$data_root/$data_name/wavs"
    fi
    if [ "$use_prep_v2" == "true" ]; then
        ./local/prep/create_decode_datav2.sh --data_root $data_root --test_sets "$data_name" --corpus_path $corpus_path
    else
        ./local/prep/create_decode_data.sh --data_root $data_root --test_sets "$data_name" --corpus_path $corpus_path
    fi
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    if [ "$skip_resample" != "true" ]; then
        python local/prep/repair_and_resample.py --data_dir $data_root/$data_name
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ./local/e2e_stt/extract_feats_whisper.sh --stage $feats_stage --stop-stage $feats_stop_stage \
                                            --data_root $data_root --data_sets $data_name \
                                            --model_name $model_name --model_tag "$model_tag" \
                                            --use_streaming $use_streaming --lang $lang --extra_options "$extra_options" \
                                            --use_whisperx $use_whisperx
    
    dest_dir=$data_root/$data_name/$model_name
    
    if [ $replace_text == "true" ]; then
        cp $dest_dir/text $data_root/$data_name/text
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    dest_dir=$data_root/$data_name/$model_name
    ./local/prep/prepare_xlsx.sh --data_root $data_root --data_name $data_name --dest_dir $dest_dir
fi
