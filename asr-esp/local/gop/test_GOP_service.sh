#!/bin/bash

trans_file="data/L2_test_small/text"
wav_file="data/L2_test_small/wav.scp"
trans_list=()
wav_list=()
id_list=()
# get wav path
while read line
do      
        IFS=' ' read -a array <<< "$line"
        wav_list+=("${array[1]}")
        id_list+=("${array[0]}")

done < "$wav_file"
# echo "${wav_list[0]}"
# echo "${id_list[0]}"

# get transcript text
while read line
do      
        IFS=' ' read -a array <<< "$line"
        trans=""
        for word in "${array[@]/${array[0]}}"
        do
                trans+="${word} "
        done
        trans_list+=("$trans")
done < "$trans_file"
# echo "${trans_list[0]}"

for ((i=0;i<${#wav_list[@]};i++))
do
        get_data=false
        python local/gop/test_GOP_service.py --uri wss://smil.empowerchinese.net:5555/client/ws/speech \
                        --rate 32000 \
                        --prompt "${trans_list[$i]}" \
                        --content-type "audio/x-raw,+layout=(string)interleaved,+rate=(int)16000,+format=(string)S16LE,+channels=(int)1" \
                        --id "${id_list[$i]}"\
                        ${wav_list[$i]}
done
