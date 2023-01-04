#!/usr/bin/env bash

data_dir="data/gept_b1"
model_name="multi_en_mct_cnn_tdnnf_tgt3meg-dl"
part=3 # 1=朗讀, 2=回答問題, 3=看圖敘述
score_names="pronunciation"
aspect=2 # 1=內容, 2=音韻, 3=詞語
stage=0
stop_stage=10000

. ./path.sh

extra_options="--do_round"

set -euo pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    exp_tag=multivar_linear_regression
    exp_root=exp/gept-p${part}
    
    if [ ! -d $exp_root/$exp_tag ]; then
        mkdir -p $exp_root/$exp_tag
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --exp_root $exp_root/$exp_tag $extra_options > $exp_root/$exp_tag/results.log
    
    python local/visualization.py   --result_root $exp_root/$exp_tag \
                                    --scores "$score_names"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    n_resamples=100
    exp_tag=multivar_linear_regression_${n_resamples}
    exp_root=exp/gept-p${part}
    
    if [ ! -d $exp_root/$exp_tag ]; then
        mkdir -p $exp_root/$exp_tag
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root/$exp_tag $extra_options > $exp_root/$exp_tag/results.log
    
    python local/visualization.py   --result_root $exp_root/$exp_tag \
                                    --scores "$score_names"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    exp_tag=multivar_linear_regression-sample_weight
    exp_root=exp/gept-p${part}
    extra_options="$extra_options --do_sample_weight"
    
    if [ ! -d $exp_root/$exp_tag ]; then
        mkdir -p $exp_root/$exp_tag
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --exp_root $exp_root/$exp_tag $extra_options > $exp_root/$exp_tag/results.log
    
    python local/visualization.py   --result_root $exp_root/$exp_tag \
                                    --scores "$score_names"
fi

exit 0
