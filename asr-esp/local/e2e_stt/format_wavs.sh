#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.


# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=wav    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Ngram model related
use_ngram=false
ngram_num=3

# Language model related
use_lm=true       # Use language model for ASR decoding.
                  # If this option is specified, lm_tag is ignored.
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
asr_tag=       # Suffix to the result dir for asr model training.
asr_exp=       # Specify the directory path for ASR experiment.
               # If this option is specified, asr_tag is ignored.
asr_stats_dir= # Specify the directory path for ASR statistics.
asr_config=    # Config for asr model training.
asr_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in asr config.
pretrained_model=              # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus.

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
k2_ctc_decoding=true
use_nbest_rescoring=true # use transformer-decoder
                         # and transformer language model for nbest rescoring
num_paths=1000 # The 3rd argument of k2.random_paths.
nll_batch_size=100 # Affect GPU memory usage when computing nll
                   # during nbest rescoring
k2_config=./conf/decode_asr_transformer_with_k2.yaml

use_streaming=false # Whether to use streaming decoding

batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_ngram=${ngram_num}gram.bin
inference_asr_model=valid.acc.ave.pth # ASR model path for decoding.
                                      # e.g.
                                      # inference_asr_model=train.loss.best.pth
                                      # inference_asr_model=3epoch.pth
                                      # inference_asr_model=valid.acc.best.pth
                                      # inference_asr_model=valid.loss.ave.pth

# [Task dependent] Set the datadir name created by local/data.sh
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

[ -z "${test_sets}" ] && { log "Error: --test_sets is required"; exit 2; };


# Check required arguments
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   log "Stage 3: Format wav.scp: data"
   # ====== Recreating "wav.scp" ======
   # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
   # shouldn't be used in training process.
   # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
   # and it can also change the audio-format and sampling rate.
   # If nothing is need, then format_wav_scp.sh does nothing:
   # i.e. the input file format and rate is same as the output.
   for dset in ${test_sets}; do
        new_dset=data/${dset}_wav
        utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${new_dset}"
        rm -f ${new_dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
        _opts=
        if [ -e data/"${dset}"/segments ]; then
            # "segments" is used for splitting wav files which are written in "wav".scp
            # into utterances. The file format of segments:
            #   <segment_id> <record_id> <start_time> <end_time>
            #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
            # Where the time is written in seconds.
            _opts+="--segments data/${dset}/segments "
        fi
        # shellcheck disable=SC2086
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
            "data/${dset}/wav.scp" "${new_dset}"

        echo "${feats_type}" > "${new_dset}/feats_type"
    done
    fi
