#!/bin/bash

dataset_names="HIV"
declare -a model_names=("GCN" "GAT" "Weave" "MPNN" "AttentiveFP" "gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking" "NF")
#declare -a model_names=("gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking")

for model_name in "${model_names[@]}"
do
    python scripts/classification_train.py -c ./datasets/MoleculeNet/HIV/split/scaffold/train_balanced.csv -sc smiles -s random -mo $model_name -p classification_results/$dataset_names/$model_name -ne 20 
done


