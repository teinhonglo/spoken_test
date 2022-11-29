#!/usr/bin/env bash

data_dir="data/gept_b1"
model_name="multi_en_mct_cnn_tdnnf_tgt3meg-dl"
part=3 # 1=朗讀, 2=回答問題, 3=看圖敘述
aspect=2 # 1=內容, 2=音韻, 3=詞語
stage=0
stop_stage=0

. ./path.sh

set -euo pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    exp_root=exp/gept-p${part}/multivar_linear_regression/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --exp_root $exp_root > $exp_root/results.log
fi

exit 0;

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    exp_dir=exp/random_forest_classifier/${exp_affix}
    if [ ! -d $exp_dir ]; then
        mkdir -p $exp_dir
    fi
    
    python local/stats_models/random_forest_classifier.py --data_dir $data_dir \
                                                          --model_name $model_name \
                                                          --part $part \
                                                          --aspect $aspect \
                                                          --exp_dir $exp_dir > $exp_dir/results.log
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    exp_dir=exp/feedforward_network_classifier/${exp_affix}
    if [ ! -d $exp_dir ]; then
        mkdir -p $exp_dir
    fi
    
    python local/stats_models/feedforward_network_classifier.py --data_dir $data_dir \
                                                                --model_name $model_name \
                                                                --part $part \
                                                                --aspect $aspect \
                                                                --exp_dir $exp_dir > $exp_dir/results.log
fi
