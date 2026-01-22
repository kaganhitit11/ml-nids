# Literature Search Guide for Final Report
## ML-NIDS Label Poisoning Project

Date: 2026-01-20

This document provides **search keywords, queries, and citation categories** to help you find and organize relevant papers for your final report. The categories align with the narrative structure of your project.

---

## 1) Search Keywords by Topic Area

### 1.1 Data Poisoning & Label Flipping Attacks (Core Topic)

**General searches:**
- "data poisoning attacks machine learning"
- "label flipping adversarial training"
- "backdoor attacks neural networks"
- "training-time attacks classification"
- "poisoning threat model"
- "Byzantine training"

**Specific attack types:**
- "class-hiding attacks" OR "class imbalance attacks"
- "feature-targeted poisoning"
- "disagreement-based attacks" OR "query by committee"
- "temporal anomaly detection"
- "loss-aware poisoning"

**Key authors/groups to search:**
- Biggio et al. (poisoning attacks pioneer)
- Wang et al. (threats to training survey)
- Chang et al. (fast adversarial label flipping)
- Koh & Liang (influence functions)

### 1.2 Defenses & Robustness (Defense Component)

**General defenses:**
- "certified defenses data poisoning"
- "outlier detection training data"
- "sample reweighting defense"
- "data cleaning machine learning"
- "robust training poisoned data"

**Specific defense mechanisms:**
- "loss-driven filtering"
- "confidence-based detection"
- "cross-validation based defense"
- "ensemble disagreement defense"

**Key authors:**
- Steinhardt et al. (certified defenses for data poisoning)
- Shejwalkar & Hoover (manipulating SGD)
- Carlini et al. (towards evaluating robustness)

### 1.3 Network Intrusion Detection Systems (NIDS Context)

**General NIDS:**
- "machine learning intrusion detection"
- "neural networks network security"
- "deep learning anomaly detection"
- "traffic classification machine learning"

**NIDS survey/benchmark:**
- "NIDS benchmark datasets"
- "intrusion detection dataset survey"
- "UNSW-NB15 evaluation"
- "CIC-IDS2017"
- "CIDDS-001"
- "CUPID dataset"

**Specific systems:**
- "flow-based intrusion detection"
- "packet-based IDS"
- "concept drift network security"

### 1.4 Adversarial Robustness (Broader Context)

**General robustness:**
- "adversarial examples neural networks"
- "robustness evaluation machine learning"
- "adversarial training defense"
- "certified robustness"

**Robustness in security:**
- "evasion attacks machine learning"
- "adversarial examples security applications"
- "robustness security systems"

### 1.5 Cross-Dataset Evaluation & Generalization

**Generalization across datasets:**
- "cross-dataset evaluation machine learning"
- "model generalization different domains"
- "dataset shift concept drift"
- "domain adaptation intrusion detection"

**Benchmark comparison:**
- "benchmark dataset construction"
- "evaluation methodology machine learning"
- "reproducibility machine learning experiments"

---

## 2) Suggested Citation Categories & Papers

Organize your bibliography into these sections:

### Section A: Foundational Work on Data Poisoning
These provide the threat model and baseline attack strategies.

**Search suggestions:**
- Biggio et al. (2012) - "Poisoning Attacks against Support Vector Machines"
- Steinhardt et al. (2017) - "Certified Defenses for Data Poisoning Attacks"
- Wang et al. (2023) - "Threats to Training: A Survey of Poisoning Attacks and Defenses"

**What to cite:**
- Threat model definition (bounded label-flip budget)
- Attack taxonomy (targeted vs. random)
- Defense taxonomy

### Section B: Specific Attack Strategies
These match your poisoning implementations.

**Class-hiding / label-flipping:**
- Search: "class hiding attack" + "label flipping"
- Look for: selective attack strategies, class imbalance exploitation

**Loss-aware / confidence-based attacks:**
- Koh & Liang (2017) - "Understanding Black-box Predictions via Influence Functions"
- Search: "high-loss sample selection" OR "low-confidence sample targeting"

**Disagreement-based attacks:**
- Seung et al. (1992) - "Query by Committee" (foundational)
- Search: "ensemble disagreement" OR "query by committee active learning"

**Temporal attacks:**
- Search: "temporal anomaly" OR "seasonal pattern attack" OR "time-based poisoning"
- Yang et al. (2021) - "CADE: Detecting and Explaining Concept Drift Samples for Security Applications"

### Section C: Defense Mechanisms
These correspond to your removal/reweighting defenses.

**Filtering/removal defenses:**
- Shejwalkar & Hoover (2021) - "Manipulating SGD with Data Ordering Attacks"
- Search: "outlier detection training" OR "suspicious sample filtering"

**Reweighting defenses:**
- Search: "sample reweighting" OR "importance weighting defense"
- Cui et al. (2019) - "Class-Balanced Loss Based on Effective Number of Samples"

**Loss/confidence-based detection:**
- Search: "cross-validation based anomaly detection training"
- Look for: detection of mislabeled/poisoned samples using model outputs

### Section D: NIDS & Intrusion Detection
Contextualizes your application domain.

**Overview / surveys:**
- Sommer & Paxson (2010) - "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection"
- Goldschmidt & Chudá (2025) - "Network Intrusion Datasets: A Survey, Limitations, and Recommendations"

**Dataset papers:**
- Moustafa & Slay (2015) - "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems"
- Rosay et al. (2021) - "From CIC-IDS2017 to LYCOS-IDS2017: A Corrected Dataset for Better Performance"
- Lawrence et al. (2022) - "CUPID: A Labeled Dataset with Pentesting for Evaluation of Network Intrusion Detection"
- Ring et al. (2017) - "Flow-based Benchmark Data Sets for Intrusion Detection" (CIDDS)

**NIDS with ML:**
- Search: "deep learning intrusion detection" OR "neural network IDS"
- Look for: recent NIDS systems using CNN, RNN, Transformer

### Section E: Robustness & Evaluation in Security
Provides broader context for robustness evaluation.

**Adversarial robustness:**
- Carlini & Wagner (2017) - "Towards Evaluating the Robustness of Neural Networks"
- Goodfellow et al. (2014) - "Explaining and Harnessing Adversarial Examples"

**Security-specific robustness:**
- Papernot et al. (2016) - "Towards the Science of Security and Privacy in Machine Learning"
- Search: "adversarial machine learning" OR "security of machine learning"

**Evaluation methodology:**
- Search: "evaluation framework machine learning security" OR "benchmark methodology"

### Section F: Related Work (Poisoning in Other Domains)
Shows that poisoning is not unique to NIDS.

**Poisoning in computer vision:**
- Search: "backdoor attack image classification" OR "data poisoning computer vision"

**Poisoning in NLP:**
- Search: "text poisoning" OR "backdoor attacks language models"

**Poisoning in other security domains:**
- Search: "malware detection adversarial" OR "spam detection poisoning"

---

## 3) Specific Search Queries for Major Databases

Use these exact or modified queries in:
- **Google Scholar** (scholar.google.com)
- **DBLP Computer Science Bibliography** (dblp.uni-trier.de)
- **arXiv** (arxiv.org)
- **IEEE Xplore** (ieeexplore.ieee.org)
- **ACM Digital Library** (dl.acm.org)

### Query 1: Core Poisoning + NIDS Intersection
```
("data poisoning" OR "label flipping" OR "training-time attack") 
AND 
("intrusion detection" OR "network security" OR "IDS" OR "NIDS")
```

### Query 2: Specific Datasets (for methodology section)
```
("UNSW-NB15" OR "CIC-IDS2017" OR "CIDDS" OR "CUPID" OR "LYCOS")
AND
("evaluation" OR "benchmark" OR "classification")
```

### Query 3: Defense in Data Poisoning
```
("defense" OR "detection" OR "mitigation") 
AND 
("data poisoning" OR "backdoor" OR "label flipping")
AND
("deep learning" OR "neural network" OR "machine learning")
```

### Query 4: Cross-Dataset Generalization
```
("cross-dataset" OR "domain shift" OR "out-of-distribution")
AND
("intrusion detection" OR "network security")
```

### Query 5: Loss-Based / Confidence-Based Attacks
```
("loss-aware" OR "confidence-based" OR "high-loss")
AND
("poisoning" OR "adversarial attack")
```

---

## 4) Recommended Starting Points (Recent Key Papers)

If you want to **get started quickly**, prioritize these:

**Must-read (directly relevant):**
1. Wang et al. (2023) - *Threats to Training survey* - gives comprehensive taxonomy
2. Steinhardt et al. (2017) - *Certified Defenses* - foundational defense work
3. Moustafa & Slay (2015) - *UNSW-NB15* - dataset context

**Important for your attack strategies:**
4. Koh & Liang (2017) - *Influence Functions* - background for loss-aware attacks
5. Seung et al. (1992) - *Query by Committee* - background for disagreement-based

**Important for NIDS context:**
6. Sommer & Paxson (2010) - *Outside the Closed World* - foundational critique of ML-IDS
7. Goldschmidt & Chudá (2025) - *NIDS Datasets Survey* - recent comprehensive overview

**Recent complementary work:**
8. Chang et al. (2023) - *Fast Adversarial Label Flipping* - similar threat model
9. Rosay et al. (2021) - *LYCOS-IDS2017* - dataset quality/correction

---

## 5) How to Organize Your Final Report Bibliography

**Suggested structure:**

```
## References

### Attacks on Machine Learning & Data Poisoning
[Papers on poisoning, backdoors, label flipping]

### Defenses & Robustness
[Papers on certified defenses, detection, filtering]

### Network Intrusion Detection Systems
[NIDS surveys, dataset papers, ML-based IDS systems]

### Adversarial Robustness & Security
[Broader adversarial ML, security evaluation]

### Evaluation Methodology & Benchmarks
[Cross-dataset evaluation, benchmark construction]
```

---

## 6) Tools for Managing Bibliography

**BibTeX management:**
- Use **JabRef** (free, open-source) to manage `.bib` files
- Export from Google Scholar, DBLP, arXiv in BibTeX format
- Check your `sample.bib` for formatting examples

**Citation helpers:**
- **Zotero** (zotero.org) - free reference manager with browser plugin
- **Mendeley** - free version available
- **Overleaf** - directly integrates BibTeX

---

## 7) Example BibTeX Entries to Add

Here are templates for key papers (fill in details from original sources):

```bibtex
% Wang et al. 2023 - Threats to Training Survey
@article{wang2023threats,
  author  = {Wang, Zeyan and others},
  title   = {Threats to Training: {A} {S}urvey of {P}oisoning {A}ttacks and {D}efenses on {M}achine {L}earning {S}ystems},
  journal = {ACM Computing Surveys},
  year    = {2023},
  volume  = {55},
  number  = {7},
  doi     = {10.1145/3538707}
}

% Steinhardt et al. 2017 - Certified Defenses
@inproceedings{steinhardt2017certified,
  author    = {Steinhardt, Jacob and Koh, Pang Wai and Liang, Percy},
  title     = {Certified {D}efenses for {D}ata {P}oisoning {A}ttacks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

% Sommer & Paxson 2010 - Foundational IDS/ML Critique
@inproceedings{sommer2010outside,
  author    = {Sommer, Robin and Paxson, Vern},
  title     = {Outside the {C}losed {W}orld: {O}n {U}sing {M}achine {L}earning for {N}etwork {I}ntrusion {D}etection},
  booktitle = {IEEE Symposium on Security and Privacy},
  year      = {2010},
  pages     = {305--316},
  doi       = {10.1109/SP.2010.25}
}

% Koh & Liang 2017 - Influence Functions
@inproceedings{koh2017understanding,
  author    = {Koh, Pang Wei and Liang, Percy},
  title     = {Understanding {B}lack-box {P}redictions via {I}nfluence {F}unctions},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  arxiv     = {1703.04730}
}

% Goldschmidt & Chudá 2025 - NIDS Datasets Survey (NEW)
@article{goldschmidt2025survey,
  author  = {Goldschmidt, P. and Chud\'a, D.},
  title   = {Network {I}ntrusion {D}atasets: {A} {S}urvey, {L}imitations, and {R}ecommendations},
  journal = {arXiv preprint},
  year    = {2025},
  arxiv   = {2502.06688}
}

% Chang et al. 2023 - Fast Adversarial Label Flipping
@article{chang2023fastflip,
  author  = {Chang, X. and Dobbie, G. and Wicker, J.},
  title   = {Fast {A}dversarial {L}abel-{F}lipping {A}ttack on {T}abular {D}ata},
  journal = {arXiv preprint},
  year    = {2023},
  arxiv   = {2310.10744}
}
```

---

## 8) Suggested Reading Order (for background building)

**Week 1: Background & Context**
- Sommer & Paxson (2010) - understand why ML-IDS is hard
- Goldschmidt & Chudá (2025) - understand dataset landscape
- Moustafa & Slay (2015) - understand UNSW-NB15 dataset

**Week 2: Core Poisoning Concepts**
- Wang et al. (2023) - comprehensive taxonomy of attacks/defenses
- Steinhardt et al. (2017) - understand certified defense framework

**Week 3: Your Specific Attacks**
- Koh & Liang (2017) - loss-aware / influence-based attacks
- Seung et al. (1992) - disagreement-based selection
- Chang et al. (2023) - label flipping methodology

**Week 4: Related Work**
- Rosay et al. (2021) - LYCOS dataset quality
- Lawrence et al. (2022) - CUPID dataset & pentesting realism
- Ring et al. (2017) - CIDDS dataset & flow-based evaluation

---

## 9) Topics NOT Yet Covered (If You Want Broader Scope)

If your final report includes sections on these, search for:

**Concept drift in security:**
- "concept drift intrusion detection"
- "non-stationary data security"

**Interpretability / explainability in security:**
- "explainable AI security" OR "interpretable intrusion detection"

**Graph-based anomaly detection:**
- "graph neural networks anomaly detection"

**Real-world deployment challenges:**
- "machine learning deployment security" OR "MLOps security"

---

## 10) Quick Reference: Your Proposal's Key Citations

Papers **already cited in your proposal** (ensure you have these):
- Sommer & Paxson (2010)
- Goldschmidt & Chudá (2025)
- Moustafa & Slay (2015) - UNSW-NB15
- Rosay et al. (2021) - LYCOS-IDS2017
- Lawrence et al. (2022) - CUPID
- Ring et al. (2017) - CIDDS
- Wang et al. (2023) - Threats to Training
- Chang et al. (2023) - Fast Flip
- Koh & Liang (2017) - Influence Functions
- Seung et al. (1992) - Query by Committee
- Yang et al. (2021) - CADE (Concept Drift)
- Steinhardt et al. (2017) - Certified Defenses

**Action item:** Ensure all of these are in your `.bib` file with complete metadata.

---

## Appendix: Search String Templates

Feel free to adapt these for your database:

```
Template 1 (Exact topic):
("data poisoning" OR "label poisoning") 
AND 
("intrusion detection" OR "NIDS")

Template 2 (Broad robustness):
("adversarial" OR "robust*")
AND
("machine learning" OR "deep learning")
AND
("defense" OR "attack")

Template 3 (Datasets):
("benchmark" OR "dataset")
AND
("network" OR "security")
AND
("intrusion" OR "anomaly")

Template 4 (Defense mechanisms):
("filtering" OR "detection" OR "defense")
AND
("poisoned" OR "corrupted")
AND
("training" OR "learning")
```

---

## End of Literature Search Guide

**Next steps:**
1. Use the search queries above to find 30-50 relevant papers
2. Organize them into the categories suggested in Section 2
3. Create a consolidated `.bib` file
4. Read the "must-read" papers first (Section 4)
5. Update your final report's literature review section with these references

