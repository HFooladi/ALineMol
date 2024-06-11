#!/bin/bash

dataset_category="TDC"
dataset_names="CYP3A4"
split_type="scaffold"
filename="external_test.csv"
declare -a model_names=("GCN" "GAT" "Weave" "MPNN" "AttentiveFP" "gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking" "NF")
#declare -a model_names=("GCN")

for model_name in "${model_names[@]}"
do
    for i in {1..20}
    do
        python scripts/classification_inference.py -f datasets/$dataset_category/$dataset_names/split/$split_type/$filename -sc smiles -t label -tp classification_results/$dataset_category/$dataset_names/$model_name/$i -ip classification_inference_results/$dataset_category/$dataset_names/$model_name -s  
    done
done
