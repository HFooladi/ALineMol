#!/bin/bash

dataset_category="TDC"
dataset_names="CYP1A2"
#split_type="perimeter"
#declare -a split_types=("max_dissimilarity" "perimeter")
declare -a split_types=("scaffold" "molecular_weight" "kmeans" "max_dissimilarity" "perimeter")
declare -a filenames=("train_0.csv" "train_1.csv" "train_2.csv" "train_3.csv" "train_4.csv" "train_5.csv" "train_6.csv" "train_7.csv" "train_8.csv" "train_9.csv")
#declare -a filenames=("train_0.csv")
#declare -a model_names=("GCN" "GAT" "Weave" "MPNN" "AttentiveFP" "gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking" "NF")
#declare -a model_names=("gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking")
declare -a model_names=("randomForest" "SVM" "XGB")
#declare -a model_names=("XGB")

for split_type in "${split_types[@]}"
do
    for filename in "${filenames[@]}"
    do
        for model_name in "${model_names[@]}"
        do
            python scripts/clf_train_ml.py -c ./datasets/$dataset_category/$dataset_names/split/$split_type/$filename -sc smiles -s random -mo $model_name -p classification_results/$dataset_category/$dataset_names/$split_type/$model_name
        done
    done
done

#for filename in "${filenames[@]}"
#do
#    for model_name in "${model_names[@]}"
#    do
#        python scripts/classification_train.py -c ./datasets/$dataset_category/$dataset_names/split/$split_type/$filename -sc smiles -s random -mo $model_name -p classification_results/$dataset_category/$dataset_names/$split_type/$model_name
#    done
#done
