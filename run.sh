#!/bin/bash

dataset_category="TDC"
declare -a dataset_names=("CYP1A2" "CYP2C9" "CYP2C19" "CYP2D6" "CYP3A4" "HIV" "AMES" "HERG")
declare -a split_types=("random""scaffold" "scaffold_generic" "molecular_weight" "molecular_weight_reverse" "molecular_logp" "kmeans" "max_dissimilarity")
declare -a filenames=("train_0.csv" "train_1.csv" "train_2.csv" "train_3.csv" "train_4.csv" "train_5.csv" "train_6.csv" "train_7.csv" "train_8.csv" "train_9.csv")
declare -a gnn_model_names=("GCN" "GAT" "Weave" "MPNN" "AttentiveFP" "gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking")
declare -a ml_model_names=("randomForest" "SVM" "XGB")

for dataset_name in "${dataset_names[@]}"
do
    for split_type in "${split_types[@]}"
    do
        for filename in "${filenames[@]}"
        do
            for model_name in "${gnn_model_names[@]}"
            do
                python scripts/clf_train_gnn.py \
                -c ./datasets/$dataset_category/$dataset_name/split/$split_type/$filename \
                -sc smiles \
                -s stratified_random \
                -mo $model_name \
                -p classification_results/$dataset_category/$dataset_name/$split_type/$model_name \
                --device cuda:0
            done
            for model_name in "${ml_model_names[@]}"
            do
                python scripts/clf_train_ml.py \
                -c ./datasets/$dataset_category/$dataset_name/split/$split_type/$filename \
                -sc smiles \
                -s stratified_random \
                -mo $model_name \
                -p classification_results/$dataset_category/$dataset_name/$split_type/$model_name
            done
        done
    done
done

declare -a testfilenames=("test_0.csv" "test_1.csv" "test_2.csv" "test_3.csv" "test_4.csv" "test_5.csv" "test_6.csv" "test_7.csv" "test_8.csv" "test_9.csv")

# run inference
for dataset_name in "${dataset_names[@]}"
do
    for split_type in "${split_types[@]}"
    do
        i=0
        for filename in "${testfilenames[@]}"
        do
            i=$((i+1))
            for model_name in "${gnn_model_names[@]}"
            do
                python scripts/clf_inference_gnn.py \
                -f datasets/$dataset_category/$dataset_name/split/$split_type/$filename \
                -sc smiles \
                -t label \
                -tp classification_results/$dataset_category/$dataset_name/$split_type/$model_name/$i \
                -ip classification_results/$dataset_category/$dataset_name/$split_type/$model_name/$i \
                --soft_classification 
            done
            for model_name in "${ml_model_names[@]}"
            do
                python scripts/clf_inference_ml.py \
                -f datasets/$dataset_category/$dataset_name/split/$split_type/$filename \
                -sc smiles \
                -t label \
                -tp classification_results/$dataset_category/$dataset_name/$split_type/$model_name/$i \
                -ip classification_results/$dataset_category/$dataset_name/$split_type/$model_name/$i \
                --soft_classification
            done
        done
    done
done
