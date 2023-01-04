#!/usr/bin/env bash

stage=0
data_root=data
test_sets="voice_2022"

. ./cmd.sh
. ./path.sh
. parse_options.sh

if [ $stage -le 0 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        for f in utt2spk wav.scp spk2utt utt2dur text feats.scp cmvn.scp; do
            if [ -f $data_root/$test_set/$f ]; then
                rm -rf $data_root/$test_set/$f;
            fi
        done
        
        for wav_path in `find $data_dir/wavs/ -name "*.wav"`; do
            wav_fn=`basename $wav_path | cut -d"." -f1`
            phn_no=`echo $wav_fn | cut -d"-" -f1`
            echo "$wav_fn $wav_path" >> $data_root/$test_set/wav.scp
            echo "$wav_fn $phn_no" >> $data_root/$test_set/utt2spk
            echo "$wav_fn $phn_no" >> $data_root/$test_set/text
        done
        
        utils/fix_data_dir.sh $data_root/$test_set
    done
fi
