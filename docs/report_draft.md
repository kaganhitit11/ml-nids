# HEADER

## Abstract

## Introduction

## Literature Review

[Maybe it is only with the introduction, not a seperate section?]


## Methodology


## Experiments and Results

## Discussion

## Conclusion


## References

## Appendix



# DRAFT: later from here is playground for drafting
Machine-learning–based Network Intrusion Detection Systems (NIDS) are widely deployed to detect malicious activity in high-volume networks, yet their robustness to training-time label corruption remains underexplored. In practice, adversaries may exploit data collection and labeling pipelines to introduce targeted label noise that selectively suppresses detection of high-value intrusions or distorts decision boundaries in critical regions of feature space. Despite the operational relevance of such structured label-poisoning threats, existing evaluations are often limited to single datasets or narrow model classes, leaving cross-dataset vulnerability poorly characterized.

This project will address this gap through a systematic, cross-dataset study of targeted label-poisoning attacks and practical defenses for NIDS. We will evaluate five complementary benchmarks, UNSW-NB15, LYCOS-IDS2017, CUPID, and CIDDS-001, using a representative model suite spanning linear and ensemble baselines and neural architectures (Logistic Regression, Random Forests, MLP, 1D-CNN and RNN). Under bounded label-flip budgets, we will implement structured poisoning strategies including class-hiding, feature-targeted, confidence-/loss-aware, disagreement-based and temporal-window and quantify robustness using accuracy, macro/per-class precision–recall–F1, confusion matrices, and degradation curves versus poisoning rate. 

We will further compare training-time defenses based on loss- and disagreement-driven filtering and reweighting, and systematically characterize how these defenses affect poisoned-data robustness and clean-data performance across datasets, model families, and attack strategies.


abstract:

We have systematcily undertaken a cross-dataset study of targeted label-poisoning attacks and practical defenses for machine-learning–based Network Intrusion Detection Systems (NIDS). Our goal was to prove the vulnerability of NIDS to label-poisoning attacks across multiple datasets and model architectures, and to evaluate the effectiveness of various defense mechanisms. 
