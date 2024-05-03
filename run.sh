#!/bin/bash

dataset_category="TDC"
dataset_names="CYP2C19"
split_type="sphere_exclusion"
filename="train_05.csv"
declare -a model_names=("GCN" "GAT" "Weave" "MPNN" "AttentiveFP" "gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking" "NF")
#declare -a model_names=("gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking")
#declare -a model_names=("GCN")

for model_name in "${model_names[@]}"
do
    python scripts/classification_train.py -c ./datasets/$dataset_category/$dataset_names/split/$split_type/$filename -sc smiles -s random -mo $model_name -p classification_results/$dataset_category/$dataset_names/$model_name -ne 20
done