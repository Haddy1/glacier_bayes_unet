#!/bin/bash

iterations=20

orig_path="dataset_16_06"
threshold=0.0025


data_path="$orig_path"

for i in `seq 1 $iterations` ; do
    out_path="$orig_path"_025_"$i"

    if [ ! -f "$data_path"/model_unet_Enze19_bayes.h5 ]; then
        break
    fi
    python3 bayes_auto_train.py --model "$data_path"/model_unet_Enze19_bayes.h5 --data_path $data_path --out_path $out_path --threshold $threshold --batch_size 25
    python3 main.py  --predict 0 --model "$data_path"/model_unet_Enze19_bayes.h5  --out_path $out_path --data_path $out_path --batch_size 25
    data_path=$out_path
done

python3 predict_bayes_train.py --basepath "$orig_path"_025


data_path="$orig_path"
threshold=0.005

for i in `seq 1 $iterations` ; do
    out_path="$orig_path"_05_"$i"

    if [ ! -f "$data_path"/model_unet_Enze19_bayes.h5 ]; then
        break
    fi
    python3 bayes_auto_train.py --model "$data_path"/model_unet_Enze19_bayes.h5 --data_path $data_path --out_path $out_path --threshold $threshold --batch_size 25
    python3 main.py  --predict 0 --model "$data_path"/model_unet_Enze19_bayes.h5  --out_path $out_path --data_path $out_path --batch_size 25
    data_path=$out_path
done

python3 predict_bayes_train.py --basepath "$orig_path"_05

