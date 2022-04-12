#!/usr/bin/env bash

set -e -o pipefail
set -x
cmd=queue.pl
#local/score_online.sh --cmd $cmd "$@"
steps/score_kaldi.sh "$@"

echo "$0: Done"
