from tqdm import tqdm

anno_fn = "data/ccs2020-21/text.anno"
pred_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid/text"
valid_fn = "data/ccs2020-21/multi_hanyu-s5-cnn1a_mct_test_valid/utt2spk"

anno_dict = {}
pred_dict = {}

valid_text_id = []

with open(anno_fn, "r") as fn:
    for line in fn.readlines():
        text_id, anno_res = line.split()
        anno_dict[text_id] = anno_res

with open(pred_fn, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        text_id = info[0]
        
        res_len = 0
        if len(info) > 1:
            res_len = len(info[1:])

        pred_dict[text_id] = res_len

for text_id, res_len in pred_dict.items():
    # 異常情況：不是單字詞 (可能有背景人聲，或重複念)
    if res_len > 1:
        continue
    # 異常情況：沒有聲音，但仍然按對
    if res_len == 0 and anno_dict[text_id] == "O":
        continue

    valid_text_id.append(text_id)


print(len(anno_dict))
print(len(pred_dict))
print(len(valid_text_id))

with open(valid_fn, "w") as fn:
    for text_id in valid_text_id:
        fn.write(text_id + " " + text_id + "\n")
