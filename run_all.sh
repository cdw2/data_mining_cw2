#!/usr/bin/bash

csv_folder="fer2018/csv"
arff_folder="fer2018/arffs"
reduction_folder="fer2018/reduced_arffs"
transformed_arffs="fer018/transformed_arffs"

cwd=$(pwd)

rm $arff_folder/*

#Convert to ARFF
for file in $csv_folder/*; do
    echo "$file";
    if [ "$file" = "$csv_folder/fer2018.csv" ];then
    ./csv_arff.py "$file"
    else
    ./csv_arff.py --skip "$file"
    fi
done;

#Reduce Data
for file in $arff_folder/*; do
    echo "$file";
    ./part3_attributeReduction.py "$file"
done;

#extract 70 pixels
echo "Starting Pixel Extraction:";

./extract_pixels.py pixel_values/pixel_values_14.txt
./extract_pixels.py pixel_values/pixel_values_35.txt
./extract_pixels.py pixel_values/pixel_values_70.txt

echo "Pixel Extraction Complete for all files";


echo "Creating Test/Training Files:";
cd fer2018/transformed_arffs
python3 TestTrain.py transformed_14.arff
python3 TestTrain.py transformed_35.arff
python3 TestTrain.py transformed_70.arff
python3 TestTrain.py fer2018.arff
mv Testtransformed* ../split/
mv Training* ../split/
mv Testfer* ../split/
echo "Done"
cd "$cwd"


#run weka
