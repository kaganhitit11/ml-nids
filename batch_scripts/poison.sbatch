#!/bin/bash
#SBATCH --job-name=poison-data     
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=4
#SBATCH --partition=ai       
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:0        
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

# python3 poisoning.py --dataset nusw --strategy confidence_based --model_type mlp --seed 42
# python3 poisoning.py --dataset nusw --strategy disagreement --model_type mlp --seed 42 --seed2 24
# python3 poisoning.py --dataset cidds --strategy confidence_based --model_type mlp --seed 42
# python3 poisoning.py --dataset cidds --strategy disagreement --model_type mlp --seed 42 --seed2 24
python3 poisoning.py --dataset cidds --strategy class_hiding
python3 poisoning.py --dataset cidds --strategy feature_predicate
python3 poisoning.py --dataset cic --strategy confidence_based --model_type mlp --seed 42
python3 poisoning.py --dataset cic --strategy disagreement --model_type mlp --seed 42 --seed2 24
python3 poisoning.py --dataset cic --strategy class_hiding
python3 poisoning.py --dataset cic --strategy feature_predicate


