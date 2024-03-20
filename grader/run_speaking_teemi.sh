#!/bin/bash
# Train from scratch
stage=0
stop_stage=1000
# data-related
score_names="content pronunciation vocabulary"
kfold=5
test_on_valid="true"
trans_type="trans_stt"
model_name=multi_en_mct_cnn_tdnnf_tgt3meg-dl
# model-related
exp_tag=
max_score=8
init_lr=5.0e-5
loss_type="cross_entropy"
test_book=1
part=1 # 1 = 基礎聽答, 2 = 情境式提問與問答, 3 = 主題式口說任務, 4 = 摘要報告 (不自動評分) 
do_split=true
do_dig=true
ori_all_bins="1,2,2.5,3,3.5,4,4.5,5"
all_bins="1.5,2.5,3.5,4.5,5.5,6.5,7.5"
cefr_bins="1.5,3.5,5.5,7.5"
sampling=default
data_prefix=
extra_options=

. ./path.sh
. ./parse_options.sh

set -euo pipefail

folds=`seq 1 $kfold`

feats_path=data/pretest/2023_teemi_tb${test_book}p${part}/$model_name/all.xlsx
data_root=data-speaking/teemi-tb${test_book}p${part}/${data_prefix}${trans_type}
exp_root=exp-speaking/teemi-tb${test_book}p${part}/${data_prefix}${trans_type}
#exp_tag=multivar_linear_regression
exp_tag=gradient_boosting_regressor_${sampling}

if [ "$test_on_valid" == "true" ]; then
    data_root=${data_root}_tov
    exp_root=${exp_root}_tov
fi

if [ "$do_dig" == "true" ]; then
    # [0, 1, 1.5, 2, 2.78, 3.5, 4, 4.25, 5, 4.75] -> [0, 1, 2, 3, 4, 6, 7, 7, 9, 8]
    echo "digitalized"
else
    all_bins=$ori_all_bins
    cefr_bins="2,3,4,5"
    data_root=${data_root}_wod
    exp_root=${exp_root}_wod
fi

if [ "$do_split" == "true" ]; then
    # 一個音檔當一個
    echo "do split"
else
    data_root=${data_root}_nosp
    exp_root=${exp_root}_nosp
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then  
    for sn in $score_names; do
        for fd in $folds; do
            data_dir=$data_root/$fd
            exp_dir=$exp_root/$exp_tag/$sn/$fd

            python local/stats_models_fd/gradient_boosting_regressor.py \
                                      $extra_options \
                                      --corpus teemi \
                                      --feats_path $feats_path \
                                      --data_dir $data_dir \
                                      --score_name $sn \
                                      --sampling $sampling \
                                      --exp_dir $exp_dir
        done
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then  
    data_dir=$data_root
    runs_root=$exp_root
    python local/speaking_predictions_to_report.py  --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report.log
    
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then  
    data_dir=$data_root
    runs_root=$exp_root
    echo $runs_root/$exp_tag
    python local/visualization.py   --result_root $runs_root/$exp_tag \
                                    --scores "$score_names"
fi

# evaluation on speaker
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then  
    data_dir=$data_root
    runs_root=$exp_root
    python local/speaking_predictions_to_report.py  --merged_speaker --data_dir $data_dir \
                                                    --result_root $runs_root/$exp_tag \
                                                    --all_bins "$all_bins" \
                                                    --cefr_bins "$cefr_bins" \
                                                    --folds "$folds" \
                                                    --question_type tb${test_book}p${part} \
                                                    --scores "$score_names" > $runs_root/$exp_tag/report_spk.log
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then  
    data_dir=$data_root
    runs_root=$exp_root
    echo $runs_root/$exp_tag
    python local/visualization.py   --result_root $runs_root/$exp_tag \
                                    --affix "_spk" \
                                    --scores "$score_names"
fi
