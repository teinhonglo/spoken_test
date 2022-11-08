
. ./path.sh

if [ ! -d steps ]; then
    ln -s $MAIN_ROOT/egs/wsj/asr1/steps .
fi

if [ ! -d utils ]; then
    ln -s $MAIN_ROOT/egs/wsj/asr1/utils .
fi

if [ ! -d pyscripts ]; then
    ln -s $MAIN_ROOT/egs2/TEMPLATE/asr1/pyscripts .
fi

if [ ! -d scripts ]; then
    ln -s $MAIN_ROOT/egs2/TEMPLATE/asr1/scripts .
fi

if [ ! -f asr.sh ]; then
    ln -s $MAIN_ROOT/egs2/TEMPLATE/asr1/asr.sh
fi

if [ ! -f db.sh ]; then
    ln -s $MAIN_ROOT/egs2/TEMPLATE/asr1/db.sh
fi
