#!/usr/bin/env bash

stage=0
stop_stage=10000
src_data_dir=data/2023_teemi
dest_data_dir=data/2023_teemi_stt
trans_root=/share/corpus/2023_teemiv2/transcript
trans_fns=

. ./cmd.sh
. ./path.sh
. parse_options.sh

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    utils/copy_data_dir.sh $src_data_dir $dest_data_dir   
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    local/teemi/prepare_text.py --trans_root $trans_root \
                                --trans_fns $trans_fns \
                                --data_dir $data_dir
fi

