#!/usr/bin/bash

csv_folder="fer2018/csv"
arff_folder="fer2018/arffs"
reduction_folder="fer2018/reduced_arffs"
transformed_arffs="fer018/transformed_arffs"

rm $arff_folder/*

for file in $csv_folder/*; do
    echo "$file";
    if [ "$file" = "$csv_folder/fer2018.csv" ];then
    ./csv_arff.py "$file"
    else
    ./csv_arff.py --skip "$file"
    fi
done;

#shuffle and split into training and test data.

#refuce

#run weka