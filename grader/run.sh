#!/usr/bin/env bash

data_dir="data/gept_b1"
model_name="multi_en_mct_cnn_tdnnf_tgt3meg-dl"
part=3 # 1=朗讀, 2=回答問題, 3=看圖敘述
score_names="pronunciation"
aspect=2 # 1=內容, 2=音韻, 3=詞語
stage=0
stop_stage=0

. ./path.sh

extra_options="--do_round"

set -euo pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    model_type=multivar_linear_regression
    exp_root=exp/gept-p${part}
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --exp_root $exp_root/$model_type $extra_options > $exp_root/results.log
    
    python local/visualization.py   --result_root $exp_root/$model_type \
                                    --scores "$score_names"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    n_resamples=100
    model_type=multivar_linear_regression_${n_resamples}
    exp_root=exp/gept-p${part}/$model_type
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root/$model_type $extra_options > $exp_root/results.log
    python local/visualization.py   --result_root $exp_root/$model_type \
                                    --scores "$score_names"
fi

exit 0

# merged below b1
affix=_bb1
extra_options="$extra_options --merge_below_b1"
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    exp_root=exp/gept-p${part}/multivar_linear_regression$affix/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --exp_root $exp_root $extra_options > $exp_root/results.log
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    n_resamples=20
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}$affix/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root $extra_options > $exp_root/results.log
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    n_resamples=50
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}$affix/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root $extra_options > $exp_root/results.log
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    n_resamples=100
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}$affix/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root $extra_options > $exp_root/results.log
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    n_resamples=150
    exp_root=exp/gept-p${part}/multivar_linear_regression_${n_resamples}$affix/
    
    if [ ! -d $exp_root ]; then
        mkdir -p $exp_root
    fi
    
    python local/stats_models/multivar_linear_regression.py --data_dir $data_dir \
                                                            --model_name $model_name \
                                                            --part $part \
                                                            --aspect $aspect \
                                                            --n_resamples $n_resamples \
                                                            --exp_root $exp_root $extra_options > $exp_root/results.log
fi
exit 0
