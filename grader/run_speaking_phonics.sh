#!/bin/bash
# Train from scratch
stage=0
stop_stage=1000
# data-related
score_names="fluency" # accuracy
kfold=5
# model-related
exp_tag=
max_score=8
init_lr=5.0e-5
loss_type="cross_entropy"
test_book=1
do_split=true
do_dig=true
# fluency
ori_all_bins="0.5,1.5,2.5,3.5"
all_bins="0.5,1.5,2.5,3.5"
cefr_bins="0.5,1.5,2.5,3.5"
# accuracy
#ori_all_bins="0.5,1.5,2.5,3.5,4.5,5.5"
#all_bins="0.5,1.5,2.5,3.5,4.5,5.5"
#cefr_bins="0.5,1.5,2.5,3.5,4.5,5.5"
labels="0,1,2,3,4"

sampling=default
data_prefix=
extra_options=

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`

data_root=data-speaking/phonics/phonics_read_aloud_all
exp_root=exp-speaking/phonics/phonics_read_aloud_all_${score_names}
exp_tag=multivar_linear_regression
#exp_tag=gradient_boosting_regressor_${sampling}

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
    for sn in $score_names; do
        for fd in $folds; do
            data_dir=$data_root/$fd
            exp_dir=$exp_root/$exp_tag/$sn/$fd

            #python local/stats_models_fd/gradient_boosting_regressor_tsv_only.py \
            python local/stats_models_fd/multivar_linear_regression_tsv_only.py \
                                      $extra_options \
                                      --corpus phonics \
                                      --data_dir $data_dir \
                                      --score_name $sn \
                                      --sampling $sampling \
                                      --exp_dir $exp_dir
        done
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then  
    data_dir=$data_root
    runs_root=$exp_root
    python local/speaking_predictions_to_report_phonics.py  --data_dir $data_dir \
                                                            --result_root $runs_root/$exp_tag \
                                                            --all_bins "$all_bins" \
                                                            --cefr_bins "$cefr_bins" \
                                                            --folds "$folds" \
                                                            --scores "$score_names" > $runs_root/$exp_tag/report.log
    
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    data_dir=$data_root
    runs_root=$exp_root
    echo $runs_root/$exp_tag
    python local/visualization_phonics.py   --result_root $runs_root/$exp_tag \
                                            --scores "$score_names" \
                                            --labels "$labels"
fi

exit 0;
