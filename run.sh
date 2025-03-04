#!/bin/bash

set -e  # Exit on error
set -u  # Exit on undefined variable

dataset_category="TDC"
declare -a dataset_names=("CYP1A2" "CYP2C9" "CYP2C19" "CYP2D6" "CYP3A4" "HIV" "AMES" "HERG")
declare -a split_types=("random" "scaffold" "scaffold_generic" "molecular_weight" "molecular_weight_reverse" "molecular_logp" "kmeans" "max_dissimilarity")
declare -a filenames=("train_0.csv" "train_1.csv" "train_2.csv" "train_3.csv" "train_4.csv" "train_5.csv" "train_6.csv" "train_7.csv" "train_8.csv" "train_9.csv")
declare -a gnn_model_names=("GCN" "GAT" "Weave" "MPNN" "AttentiveFP" "gin_supervised_contextpred" "gin_supervised_infomax" "gin_supervised_edgepred" "gin_supervised_masking")
declare -a ml_model_names=("randomForest" "SVM" "XGB")
declare -a testfilenames=("test_0.csv" "test_1.csv" "test_2.csv" "test_3.csv" "test_4.csv" "test_5.csv" "test_6.csv" "test_7.csv" "test_8.csv" "test_9.csv")


# Function to check if required directories exist
check_directories() {
    local dirs=(
        "./datasets/${dataset_category}"
        "classification_results/${dataset_category}"
        "scripts"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            echo "Error: Required directory '$dir' does not exist"
            exit 1
        fi
    done
}

# Function to check GPU availability
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found. GPU may not be available."
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        echo "Error: Unable to access GPU"
        exit 1
    fi
}

# Function to run training for a specific model type
run_training() {
    local dataset_name=$1
    local split_type=$2
    local filename=$3
    local model_type=$4
    local model_name=$5
    
    local base_cmd="python scripts/clf_train_${model_type}.py"
    local output_dir="classification_results/${dataset_category}/${dataset_name}/${split_type}/${model_name}"
    
    mkdir -p "$output_dir"
    
    echo "Running ${model_type} training for ${model_name} on ${dataset_name}/${split_type}/${filename}"
    
    if [[ "$model_type" == "gnn" ]]; then
        $base_cmd \
            -c "./datasets/${dataset_category}/${dataset_name}/split/${split_type}/${filename}" \
            -sc smiles \
            -s stratified_random \
            -mo "$model_name" \
            -p "$output_dir" \
            --device cuda:0 || {
                echo "Error: Training failed for ${model_name} on ${dataset_name}/${split_type}/${filename}"
                return 1
            }
    else
        $base_cmd \
            -c "./datasets/${dataset_category}/${dataset_name}/split/${split_type}/${filename}" \
            -sc smiles \
            -s stratified_random \
            -mo "$model_name" \
            -p "$output_dir" || {
                echo "Error: Training failed for ${model_name} on ${dataset_name}/${split_type}/${filename}"
                return 1
            }
    fi
}

# Function to run inference
run_inference() {
    local dataset_name=$1
    local split_type=$2
    local filename=$3
    local model_type=$4
    local model_name=$5
    local index=$6
    
    local base_cmd="python scripts/clf_inference_${model_type}.py"
    local output_dir="classification_results/${dataset_category}/${dataset_name}/${split_type}/${model_name}/${index}"
    
    echo "Running ${model_type} inference for ${model_name} on ${dataset_name}/${split_type}/${filename}"
    
    $base_cmd \
        -f "datasets/${dataset_category}/${dataset_name}/split/${split_type}/${filename}" \
        -sc smiles \
        -t label \
        -tp "$output_dir" \
        -ip "$output_dir" \
        --soft_classification || {
            echo "Error: Inference failed for ${model_name} on ${dataset_name}/${split_type}/${filename}"
            return 1
        }
}

main() {
    echo "Starting script at $(date)"
    
    # Check requirements
    check_directories
    check_gpu
    
    # Training phase
    echo "Starting training phase..."
    for dataset_name in "${dataset_names[@]}"; do
        for split_type in "${split_types[@]}"; do
            for filename in "${filenames[@]}"; do
                # Run GNN models
                for model_name in "${gnn_model_names[@]}"; do
                    run_training "$dataset_name" "$split_type" "$filename" "gnn" "$model_name"
                done
                
                # Run ML models
                for model_name in "${ml_model_names[@]}"; do
                    run_training "$dataset_name" "$split_type" "$filename" "ml" "$model_name"
                done
            done
        done
    done
    
    # Inference phase
    echo "Starting inference phase..."
    for dataset_name in "${dataset_names[@]}"; do
        for split_type in "${split_types[@]}"; do
            local i=0
            for filename in "${testfilenames[@]}"; do
                i=$((i+1))
                
                # Run GNN inference
                for model_name in "${gnn_model_names[@]}"; do
                    run_inference "$dataset_name" "$split_type" "$filename" "gnn" "$model_name" "$i"
                done
                
                # Run ML inference
                for model_name in "${ml_model_names[@]}"; do
                    run_inference "$dataset_name" "$split_type" "$filename" "ml" "$model_name" "$i"
                done
            done
        done
    done
    
    echo "Script completed at $(date)"
}

# Run the main function
main
