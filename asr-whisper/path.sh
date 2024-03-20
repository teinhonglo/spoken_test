export KALDI_ROOT=/share/nas167/teinhonglo/kaldis/kaldi-cu11
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/gopbin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1

for d in steps utils; do
    if [ ! -d $d ]; then
        ln -s $KALDI_ROOT/egs/wsj/s5/$d
    fi
done

# your conda (python)
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"


if [ "$use_whisperx" == "true" ]; then
    echo "Set env to whisperx"
    conda activate whisperx
else
    echo "Set env to whisper"
    conda activate whisper
fi
