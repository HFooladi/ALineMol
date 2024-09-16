#!/bin/bash

dataset_category="TDC"
dataset_names="CYP2D6"
#declare -a split_types=("max_dissimilarity")
declare -a split_types=("scaffold" "molecular_weight" "kmeans" "max_dissimilarity" "perimeter")
declare -a filenames=("test_0.csv" "test_1.csv" "test_2.csv" "test_3.csv" "test_4.csv" "test_5.csv" "test_6.csv" "test_7.csv" "test_8.csv" "test_9.csv")
#declare -a model_names=("GCN" "GAT" "Weave" "MPNN" "AttentiveFP" "gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking" "NF")
declare -a model_names=("randomForest" "SVM" "XGB")
for split_type in "${split_types[@]}"
do
    i=0
    for filename in "${filenames[@]}"
    do
        i=$((i+1))
        for model_name in "${model_names[@]}"
        do
            python scripts/clf_inference_ml.py -f datasets/$dataset_category/$dataset_names/split/$split_type/$filename -sc smiles -t label -tp classification_results/$dataset_category/$dataset_names/$split_type/$model_name/$i -ip classification_results/$dataset_category/$dataset_names/$split_type/$model_name/$i --soft_classification 
        done
    done
done