#!/usr/bin/env bash

data_dir="data/gept_b1"
model_name="multi_en_mct_cnn_tdnnf_tgt3meg-dl"
part=3 # 1=朗讀, 2=回答問題, 3=看圖敘述
aspect=2 # 1=內容, 2=音韻, 3=詞語
stage=0
stop_stage=10000

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

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    n_resamples=20
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root > $exp_root/results.log
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    n_resamples=50
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root > $exp_root/results.log
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    n_resamples=100
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root > $exp_root/results.log
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    n_resamples=150
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root > $exp_root/results.log
fi

exit 0
