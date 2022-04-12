#!/usr/bin/env bash

stage=0
data_root=data
data_name=spoken_test_2022_mar18
dest_dir=spoken_test_2022_mar18/gigaspeech

. ./cmd.sh
. ./path.sh
. parse_options.sh

# create info.xlsx trans.xlsx

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
if [ $stage -le 0 ]; then
    python local/prep/create_info_xlsx.py --data_dir $data_root/$data_name 
fi

if [ $stage -le 1 ]; then
    python local/prep/create_trans_xlsx.py --data_dir $data_root/$data_name --dest $dest_dir
fi

if [ $stage -le 1 ]; then
    python local/prep/json2xlsx.py --data_dir $dest_dir
fi

conda activate
