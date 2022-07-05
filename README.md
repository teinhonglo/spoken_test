# 0. Environment

### 0-1. 已安裝Espnet & KALDI
```
cd s5-esp
nano path.sh
# 把$MAIN_ROOT修改成$ESPNET_ROOT的路徑
ln -s $ESPNET_ROOT/tools/kaldi/egs/wsj/s5/steps
ln -s $ESPNET_ROOT/tools/kaldi/egs/wsj/s5/utils
cd -

cd s5-kaldi
nano path.sh
# 把$KLADI_ROOT修改成$KALDI_ROOT的路徑
ln -s $KALDI_ROOT/egs/wsj/s5/steps
ln -s $KALDI_ROOT/egs/wsj/s5/utils
cd -
```

# 1. Data preparation
### 1-1. data directory
```
data/spoken_test_2022_jan28
```

### 1-2. espnet
```
cd s5-esp
local/prep/prepare_data.sh --stage -2 --stop-stage -1
```

### 1-2. kaldi (alternative)
```
cd s5-kaldi
local/prep/prepare_data_kaldi.sh --stage -2 --stop-stage -1
```

# 2. Feature extraction

### 2-1. espnet
```
cd s5-esp
local/prep/prepare_data.sh --data_name data/spoken_test_2022_jan28
```

### 2-1. kaldi (alternative)
```
cd s5-kaldi
local/prep/prepare_data_kaldi.sh --data_name data/spoken_test_2022_jan28
```

# 3. Grader
```
cd grader
python local/stats_models/multivar_linear_regression.py
```

# 4. Notes
### 修改成自己conda的啟動方式
```
https://github.com/teinhonglo/dhe_spoken_test/blob/main/s5-esp/local/prep/prepare_data.sh#L28
https://github.com/teinhonglo/dhe_spoken_test/blob/main/s5-esp/local/prep/prepare_data_kaldi.sh#L29
https://github.com/teinhonglo/dhe_spoken_test/blob/main/s5-esp/local/kaldi_stt/extract_feats.sh#L150
```
