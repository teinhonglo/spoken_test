export KALDI_ROOT=/share/nas167/teinhonglo/kaldis/kaldi-cu11
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/gopbin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1
# your conda (python)
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate whisper
