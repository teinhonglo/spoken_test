export KALDI_ROOT=/share/nas165/teinhonglo/kaldi-nnet3-specaug
export WENET_ROOT=/share/nas167/teinhonglo/wenets/wenet
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/gopbin:$WENET_ROOT/runtime/server/x86/build:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONDONTWRITEBYTECODE=1
