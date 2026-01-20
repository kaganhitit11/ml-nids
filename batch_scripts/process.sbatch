#!/bin/bash
#SBATCH --job-name=train-rnn     
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --partition=ai      
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:0:0        
#SBATCH --output=logs/job_%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ohitit20@ku.edu.tr   

module load conda3/latest
module load cuda

export PYTHONNOUSERSITE=1

conda env list

conda activate ip-lora

REPO_DIR='/home/ohitit20/ml-nids'

cd $REPO_DIR

python3 split_cic.py