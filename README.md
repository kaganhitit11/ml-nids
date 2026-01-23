# Cross Dataset Evaluation of Robustness in Machine Learning Based Intrusion Detection Under Structured Label Poisoning

## Overview

Machine learning-based Network Intrusion Detection Systems (NIDS) are increasingly deployed to
protect critical infrastructure, yet their vulnerability to training-time attacks remains poorly understood.
This work presents the first comprehensive cross-dataset evaluation of label poisoning attacks against
ML-based NIDS, systematically studying how adversaries who corrupt training labels can blind intrusion
detectors while maintaining deceptively high overall accuracy, a phenomenon we term the accuracy illusion.
We evaluate five model architectures (CNN, RNN, MLP, Logistic Regression, Random Forest) across four
benchmark datasets (CIC-IDS2017, UNSW-NB15, CUPID, CIDDS-001) under five poisoning strategies at
varying intensities (5%, 10%, 20%). Our experiments reveal alarming vulnerabilities: on CIC-IDS2017,
a simple class hiding attack at 20% poisoning reduced attack recall from 98% to 0% while accuracy
remained above 80%. We find that dataset class imbalance critically determines vulnerability; balanced
datasets resist poisoning while imbalanced datasets enable complete detection failure. Surprisingly, simple
random label flipping outperforms sophisticated targeted attacks. We evaluate removal and reweighting
defenses, documenting their architecture-dependent effectiveness and ultimate failure at high poisoning
rates. We also report a significant baseline failure on CIDDS-001, where extreme class imbalance (367:1)
renders standard models non-functional as intrusion detectors even on clean data, a cautionary finding
for practitioners. Our results establish that accuracy alone cannot assess NIDS health, and we provide
concrete recommendations for robust deployment.

## Repository Structure

- `models.py` - Model architectures (CNN, RNN, MLP, Logistic Regression, Random Forest)
- `dataloaders.py` - Dataset loading and preprocessing utilities
- `poisoning.py` - Label poisoning attack implementations
- `train.py` - Training script with poisoning and defense options
- `process_cic.py`, `process_cidds.py` - Dataset preprocessing scripts
- `data/` - Processed dataset structures (train/test splits)
- `batch_scripts/` - SLURM batch scripts for HPC execution
- `docs/` - Research documentation and final report

- For simplicity, only the dataset structures are shared in this repository. If you would like to reach the datasets, please contact us.

---

Oğuz Kağan Hitit, Damla Görgülü, Eren Yavuz, Hakan Çapuk, Rana Ataseven

(ohitit20, dgorgulu21, eyavuz21, hcapuk20, rataseven21)@ku.edu.tr

Koç University, Fall 2025, COMP 430/530 Final Project
