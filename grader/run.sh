#!/usr/bin/env bash

data_dir="data/gept_b1"
model_name="multi_en_mct_cnn_tdnnf_tgt3meg-dl"
part=2 # 1=朗讀, 2=回答問題, 3=看圖敘述
score_names="content pronunciation vocabulary"
stage=0
stop_stage=10000

. ./path.sh

extra_options="--do_round"

set -euo pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    exp_tag=multivar_linear_regression
    exp_root=exp/gept-p${part}

    for score_name in $score_names; do
        exp_dir=$exp_root/$exp_tag/$score_name
        if [ ! -d $exp_dir ]; then
            mkdir -p $exp_dir
        fi
        
        python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                                --model_name $model_name \
                                                                --part $part \
                                                                --score_name $score_name \
                                                                --exp_root $exp_root/$exp_tag $extra_options > $exp_dir/results.log
        python local/visualization.py   --result_root $exp_root/$exp_tag \
                                        --scores "$score_name"
    done
    
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    n_resamples=100
    exp_tag=multivar_linear_regression_${n_resamples}
    exp_root=exp/gept-p${part}
    
    
    for score_name in $score_names; do
        exp_dir=$exp_root/$exp_tag/$score_name
        if [ ! -d $exp_dir ]; then
            mkdir -p $exp_dir
        fi
        
        python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                                --model_name $model_name \
                                                                --part $part \
                                                                --score_name $score_name \
                                                                --n_resamples $n_resamples \
                                                                --exp_root $exp_root/$exp_tag $extra_options > $exp_dir/results.log
        
        python local/visualization.py   --result_root $exp_root/$exp_tag \
                                        --scores "$score_name"
    done
    
fi

exit 0
