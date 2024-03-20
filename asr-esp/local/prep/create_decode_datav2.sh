#!/usr/bin/env bash

stage=0
corpus_path=
data_root=data
test_sets="voice_2022"

. ./cmd.sh
. ./path.sh
. parse_options.sh

echo "creating test data"    
if [ $stage -le 0 ]; then
    for test_set in $test_sets; do
        data_dir=$data_root/$test_set
        
        if [ ! -d $data_dir ]; then
            mkdir -p $data_dir
        fi

        for f in utt2spk wav.scp spk2utt utt2dur text feats.scp cmvn.scp; do
            if [ -f $data_root/$test_set/$f ]; then
                rm -rf $data_root/$test_set/$f;
            fi
        done
        
        if [ -z $corpus_path ]; then
            corput_path="$data_dir/wavs"
        fi

        for wav_path in `find $corpus_path -name "*.wav"`; do
            wav_fn=`echo $wav_path | awk -F"/" '{print $4"-"$5"-"$6"-"$7"-"$8}' | cut -d"." -f1`
            phn_no=`echo $wav_fn | cut -d"-" -f1`
            echo "$wav_fn $wav_path" >> $data_root/$test_set/wav.scp
            echo "$wav_fn $phn_no" >> $data_root/$test_set/utt2spk
            echo "$wav_fn $phn_no" >> $data_root/$test_set/text
        done
        
        utils/fix_data_dir.sh $data_root/$test_set
    done
fi
