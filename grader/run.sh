#!/usr/bin/env bash

data_dir="data/spoken_test_2022_jan28"
model_name="librispeech_mct_tdnnf_kaldi_tgt3"
part=3 # 1=朗讀, 2=回答問題, 3=看圖敘述
aspect=2 # 1=內容, 2=音韻, 3=詞語
stage=0
stop_stage=10000

. ./path.sh

set -euo pipefail


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    exp_dir=exp/linear_regression
    if [ ! -d $exp_dir ]; then
        mkdir -p $exp_dir
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --exp_dir $exp_dir > $exp_dir/results.log
fi
