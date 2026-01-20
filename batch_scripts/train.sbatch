#!/bin/bash
#SBATCH --job-name=train-rnn     
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --partition=ai       
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:0:0        
#SBATCH --output=logs/job_%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ohitit20@ku.edu.tr   

module load conda3/latest
module load cuda

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

conda env list

conda activate ip-lora

REPO_DIR='/home/ohitit20/ml-nids'

cd $REPO_DIR

# Nonpoisoned models



DATASET_NAME="cidds"
POISONING_STRATEGIES=("class_hiding" "feature_predicate" "disagreement" "confidence_based")
POISONING_RATES=("005" "010" "020")
MODEL_NAMES=("logistic" "random_forest" "mlp" "rnn" "cnn")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    python3 train.py --dataset $DATASET_NAME --model $MODEL_NAME --data_dir data_real/$DATASET_NAME/ --epochs 3
done

for POISONING_STRATEGY in "${POISONING_STRATEGIES[@]}"; do
    for POISONING_RATE in "${POISONING_RATES[@]}"; do
        for MODEL_NAME in "${MODEL_NAMES[@]}"; do
            python3 train.py --dataset $DATASET_NAME --model $MODEL_NAME --data_dir data_real/$DATASET_NAME/poisoned/$POISONING_STRATEGY/$POISONING_RATE --epochs 3
        done
    done
done


