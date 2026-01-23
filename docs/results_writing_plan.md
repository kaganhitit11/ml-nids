# Results, Discussion, and Conclusion Writing Plan

## Overview

This document provides a structured roadmap for writing Sections 5 (Results), 6 (Discussion), and 7 (Conclusion) of your final report. Follow this plan systematically to ensure comprehensive coverage of your experimental findings.

---

## Section 5: Results

**Goal:** Present objective observations from your experiments without interpretation. Report what happened, not why it happened.

### Structure

```
Section 5: Results
├── 5.1 Baseline Performance on Clean Data
├── 5.2 CIC-IDS2017 Results
│   ├── 5.2.1 Baseline and Poisoning Impact
│   ├── 5.2.2 Defense Effectiveness
│   └── 5.2.3 Summary Table
├── 5.3 UNSW-NB15 Results
│   ├── 5.3.1 Baseline and Poisoning Impact
│   ├── 5.3.2 Defense Effectiveness
│   └── 5.3.3 Summary Table
├── 5.4 CUPID Results
│   ├── 5.4.1 Baseline and Poisoning Impact
│   ├── 5.4.2 Temporal Window Poisoning
│   ├── 5.4.3 Defense Effectiveness
│   └── 5.4.4 Summary Table
├── 5.5 CIDDS-001 Results and Technical Issues
└── 5.6 Cross-Dataset Overview
```

---

### 5.1 Baseline Performance on Clean Data

**What to write:**
- Report clean baseline Macro F1, Accuracy, Attack Recall for each model on each dataset
- Use Table format for clarity

**Data sources:**
- `eval_results/*/cic_clean_0.json`
- `eval_results/*/unsw_clean_0.json`
- `eval_results/*/cupid_clean_0.json`
- `eval_results/*/cidds_clean_0.json`

**Example table structure:**
```
| Model | CIC-IDS2017 | UNSW-NB15 | CUPID | CIDDS-001 |
|-------|-------------|-----------|-------|-----------|
|       | Acc / F1 / Recall | ... | ... | ... |
| LR    | 0.XX / 0.XX / 0.XX | ... | ... | ... |
| RF    | ... | ... | ... | ... |
| MLP   | ... | ... | ... | ... |
| CNN   | ... | ... | ... | ... |
| RNN   | ... | ... | ... | ... |
```

**Key questions to answer:**
1. Which models perform best on clean data?
2. Which datasets are inherently harder (lower clean F1)?
3. What is the attack recall (TP rate) on clean data? (This is your reference point)

**⚠️ CRITICAL CHECK:** 
- For CIDDS-001, you should observe that clean Macro F1 ≈ 50% and attack recall ≈ 0-7%
- This indicates model failure even without poisoning
- Note this explicitly but don't interpret yet (save for Section 6)

---

### 5.2 CIC-IDS2017 Results

**Reference:** Figures 2-6 (second row), eval_results_table.md filtered for `dataset=cic`

#### 5.2.1 Baseline and Poisoning Impact

**What to write:**
For each poisoning strategy (class_hiding, feature_predicate, confidence_based, disagreement), report:

1. **Clean baseline** (already covered in 5.1, just reference it)
2. **Impact at 5% poisoning** - Report drop in Macro F1, Attack Recall
3. **Impact at 10% poisoning** - Report drop in Macro F1, Attack Recall
4. **Impact at 20% poisoning** - Report drop in Macro F1, Attack Recall

**Data to extract:**
From `eval_results_table.md`, filter for:
- `dataset = cic`
- `attack_type = class_hiding`, `poison_pct = 005/010/020`, `defense = (empty)`
- Record: `test_accuracy`, `macro_f1`, `pos_recall`
- Repeat for other attack types

**Example narrative:**
> Under class hiding attack on CIC-IDS2017, the 1D-CNN model degraded severely. At 5% poisoning, Macro F1 remained high at 0.96, with attack recall at 0.92. However, at 10% poisoning, Macro F1 dropped to 0.59 with attack recall collapsing to 0.16. At 20% poisoning, the model achieved 0% attack recall (TP=0) while overall accuracy remained at 80.3%.

**⚠️ CRITICAL FINDING - The Accuracy Illusion:**
Create a dedicated sub-subsection or highlighted paragraph:
> **The Accuracy Illusion:** Observe from Table X that at 20% class hiding on CIC-IDS2017, CNN reports:
> - Test Accuracy: 80.3%
> - Macro F1: 44.5%
> - Attack Recall: 0%
> - Confusion Matrix: TN=454,620, FP=0, FN=111,529, TP=0
>
> The model classifies ALL samples as benign, achieving high accuracy due to class imbalance (4:1 benign:attack), while completely failing to detect attacks.

#### 5.2.2 Defense Effectiveness

**What to write:**
For each poisoning rate and attack type, compare:
- No Defense (baseline poisoned)
- Removal Defense
- Reweighting Defense

Report whether defense restored performance, partially restored, or failed.

**Data to extract:**
From `eval_results_table.md`, for each `(attack_type, poison_pct)` combination:
- Compare `defense = (empty)` vs `defense = removal` vs `defense = reweighting`
- Report the change in Macro F1 and Attack Recall

**Example narrative:**
> At 5% class hiding, removal defense successfully restored CNN performance from F1=0.96 to F1=0.94, though slightly lower than clean baseline (0.97). However, at 20% poisoning, both removal and reweighting defenses completely failed, achieving the same 0% attack recall as the undefended poisoned model.

#### 5.2.3 Summary Table

Create a consolidated table for CIC-IDS2017:

```
| Attack Type | Rate | Model | Clean F1 | Poisoned F1 | Removal F1 | Reweighting F1 |
|-------------|------|-------|----------|-------------|------------|----------------|
| Class Hiding | 5%  | CNN   | 0.97     | 0.96        | 0.94       | 0.95          |
| Class Hiding | 10% | CNN   | 0.97     | 0.59        | 0.59       | 0.54          |
| Class Hiding | 20% | CNN   | 0.97     | 0.45        | 0.45       | 0.45          |
| ... | ... | ... | ... | ... | ... | ... |
```

Add similar rows for RF, MLP, RNN, LR and for other attack types.

---

### 5.3 UNSW-NB15 Results

**Reference:** Figures 2-6 (first row), eval_results filtered for `dataset=unsw`

Follow the same structure as 5.2:
- 5.3.1 Baseline and Poisoning Impact
- 5.3.2 Defense Effectiveness
- 5.3.3 Summary Table

**Key observation to report:**
Based on your Gemini feedback and Figure 6 (RF), UNSW-NB15 is MORE RESILIENT:
> "Even under 20% Class Hiding, the Random Forest model's performance remained stable around 82-83%"

Report this quantitatively. Why is it more resilient? Save interpretation for Section 6, but note the contrast with CIC-IDS2017.

---

### 5.4 CUPID Results

**Reference:** Figure 1, eval_results filtered for `dataset=cupid`

Follow same structure as 5.2, but add:

#### 5.4.2 Temporal Window Poisoning

This attack is unique to CUPID. Report:
- Impact at 5%, 10%, 20%
- Compare to other attack strategies on CUPID
- Does temporal targeting work better than random?

---

### 5.5 CIDDS-001 Results and Technical Issues

**⚠️ CRITICAL SECTION**

From Gemini's feedback:
> "CIDDS-001 baseline anomaly: clean Macro F1 ≈ 50%, models predicting all benign"

**What to write:**
Be honest and transparent:

> **5.5 CIDDS-001: Baseline Training Failure**
>
> The CIDDS-001 dataset exhibited a critical baseline training failure across all models. As shown in Table X, even on clean data, Macro F1 scores hovered around 50%, with attack recall consistently below 7%. Examination of confusion matrices reveals that models predict the majority class (benign) for nearly all samples.
>
> For example, the 1D-CNN on clean CIDDS-001 achieved:
> - Test Accuracy: 99.7%
> - Macro F1: 56.1%
> - Attack Recall: 6.9%
> - Confusion Matrix: TN=1,306,122, FP=171, FN=3,315, TP=247
>
> This indicates severe class imbalance (approx. 367:1) combined with insufficient training epochs (3 epochs) for this dataset. Consequently, poisoning attacks had no measurable impact (performance was already at floor). All further results for CIDDS-001 are omitted as unreliable.

**Action items:**
1. Report the failure honestly
2. Provide hypothesis (class imbalance + insufficient training)
3. State that you exclude CIDDS-001 from further analysis
4. In Discussion, mention this as a limitation and lesson learned

---

### 5.6 Cross-Dataset Overview

**What to write:**
After presenting per-dataset results, synthesize high-level patterns:

**Topics to cover:**

1. **Attack Effectiveness Ranking**
   - Which attack was most effective across datasets?
   - Report: "Class Hiding caused the largest average F1 drop (X%), followed by..."

2. **Model Vulnerability Comparison**
   - Which models were most vulnerable?
   - Report: "Neural networks (CNN, RNN, MLP) showed similar vulnerability to statistical models (LR, RF), with CNN experiencing the largest drop on CIC-IDS2017 (F1: 0.97 → 0.45 at 20% class hiding)."

3. **Dataset Resilience Ranking**
   - Order datasets by resilience: "UNSW-NB15 > CUPID > CIC-IDS2017 > CIDDS-001 (excluded)"
   - Report the numeric evidence

4. **Defense Success Rate**
   - At what poisoning rates do defenses work?
   - Report: "Defenses showed effectiveness at 5% poisoning (avg F1 recovery: X%), partial effectiveness at 10% (avg recovery: Y%), and near-zero effectiveness at 20%."

**Table suggestion:**

```
| Dataset | Most Effective Attack | Most Vulnerable Model | Avg F1 Drop at 20% |
|---------|----------------------|----------------------|--------------------|
| CIC-IDS2017 | Class Hiding | 1D-CNN | 0.52 (0.97→0.45) |
| UNSW-NB15 | Class Hiding | MLP | 0.15 (0.85→0.70) |
| CUPID | Temporal Window | RNN | 0.XX |
| CIDDS-001 | N/A (baseline failure) | N/A | N/A |
```

---

## Section 6: Discussion

**Goal:** Interpret the results. Explain WHY things happened. Connect to literature. Discuss implications.

### Structure

```
Section 6: Discussion
├── 6.1 The Accuracy Illusion
├── 6.2 The Paradox of Simple Attacks
├── 6.3 Dataset-Specific Vulnerability
├── 6.4 Model Architecture and Robustness
├── 6.5 Defense Limitations
├── 6.6 Implications for Practitioners
└── 6.7 Limitations and Lessons Learned
```

---

### 6.1 The Accuracy Illusion

**What to discuss:**
- **Observation:** Models achieve high accuracy (80%+) while detecting 0% of attacks
- **Why it happens:** Class imbalance + all predictions = benign
- **Why it's dangerous:** Practitioners monitoring accuracy dashboards will not notice the failure
- **Connection to literature:** This is exactly the "closed world" problem Sommer & Paxson warned about

**Example text:**
> The most alarming finding is what we term the "Accuracy Illusion." On CIC-IDS2017 under 20% class hiding, the CNN model maintained 80.3% test accuracy while achieving 0% attack recall. This occurs because the dataset has a 4:1 benign-to-attack ratio—a model that predicts "benign" for all traffic is 80% accurate by default.
>
> In operational settings, this is catastrophic. Security teams monitoring aggregate accuracy metrics would see 80% performance and assume the system is functional, while in reality, the NIDS has been completely blinded to malicious traffic. This validates the warnings by Sommer & Paxson (2010) about the inadequacy of accuracy as a metric in imbalanced security domains.

**Action:** Reference back to Introduction where you promised this finding. Connect the dots.

---

### 6.2 The Paradox of Simple Attacks

**What to discuss:**
- **Observation:** Class Hiding (random flipping) outperformed sophisticated attacks (feature-targeted, influence-aware)
- **Why it's counter-intuitive:** ML security literature emphasizes optimization-based attacks
- **Hypothesis:** Explain why this might happen

**Hypothesis to explore:**
1. **Sophisticated attacks target "hard" examples** near decision boundary
   - Models might treat these as noise and ignore them (robust to local perturbations)
2. **Random attacks corrupt global statistics**
   - Class hiding blurs the entire feature distribution
   - Models can't distinguish signal from noise anywhere

**Example text:**
> Contrary to expectations from adversarial ML literature, simple random label flipping (Class Hiding) proved far more effective than targeted strategies. On CIC-IDS2017, Class Hiding at 20% reduced F1 to 0.45, while Feature-Targeted poisoning only reduced it to 0.95.
>
> We hypothesize this occurs because sophisticated attacks select "boundary" samples—difficult cases already near the decision threshold. Modern ensemble methods (RF) and overparameterized neural networks can treat these as outliers and maintain reasonable decision boundaries. In contrast, Class Hiding introduces widespread noise across the entire feature manifold, fundamentally corrupting the model's learned representation of "normal" vs. "malicious" distributions.
>
> This finding challenges the assumption that adversaries must be sophisticated to be effective. Simple, untargeted attacks pose a greater threat to NIDS than previously recognized.

---

### 6.3 Dataset-Specific Vulnerability

**What to discuss:**
- **Observation:** CIC-IDS2017 highly vulnerable, UNSW-NB15 resilient
- **Why?** Explore possible explanations:
  - Feature separability
  - Class imbalance differences
  - Attack diversity
  - Feature engineering quality

**Example text:**
> Dataset choice dramatically influenced vulnerability. CIC-IDS2017 exhibited catastrophic failure under Class Hiding (F1: 0.97 → 0.45), while UNSW-NB15 maintained stability (F1: 0.85 → 0.70 under identical conditions).
>
> We attribute this to two factors:
> 1. **Class balance:** UNSW-NB15's 1:1 benign-to-attack ratio makes the accuracy illusion harder to hide. A model predicting all benign would achieve 50% accuracy, triggering alarms.
> 2. **Feature quality:** UNSW-NB15's features (extracted via Argus/Bro) may provide more robust, diverse signal compared to CIC-IDS2017's flow statistics.
>
> **Critical implication:** Conclusions drawn from a single dataset do not generalize. A study using only UNSW-NB15 would conclude NIDS are robust; one using only CIC-IDS2017 would declare them catastrophically vulnerable. Our cross-dataset methodology reveals this variance.

---

### 6.4 Model Architecture and Robustness

**What to discuss:**
- **Observation:** Neural networks not more robust than statistical models
- **Implication:** Complexity ≠ Robustness

**Example text:**
> We found no evidence that neural network architectures provide inherent robustness against label poisoning. CNN, RNN, and MLP models exhibited similar or greater vulnerability compared to Logistic Regression and Random Forest on poisoned data.
>
> This suggests that architectural inductive biases (e.g., convolutional locality, recurrent memory) do not confer adversarial robustness when the attack targets labels rather than features. All models rely on the correctness of training labels; none have built-in mechanisms to detect systematic label corruption.

---

### 6.5 Defense Limitations

**What to discuss:**
- **Observation:** Defenses work at 5%, fail at 20%
- **Why?** Theoretical and practical limits
- **Connection to literature:** Steinhardt et al.'s certified bounds

**Example text:**
> Both removal and reweighting defenses showed effectiveness at low poisoning rates (5%) but collapsed at high rates (20%). This aligns with theoretical results by Steinhardt et al. (2017), who showed that defenses relying on empirical centroids can be subverted when the adversary controls a large fraction of data.
>
> **Practical implication:** Current defenses assume poisoning is a "small noise" problem. At 20% corruption, the training distribution is fundamentally altered, and loss-based outlier detection cannot distinguish corrupted samples from legitimate difficult examples.
>
> **Assumption caveat:** Our defenses assume the defender knows the poisoning rate (oracle budget in Eq. X). In practice, this must be estimated, further degrading defense effectiveness.

---

### 6.6 Implications for Practitioners

**What to discuss:**
Translate findings into actionable advice.

**Recommendations:**
1. **Do not rely on accuracy alone** - Monitor per-class recall, especially for attack class
2. **Validate training data provenance** - Understand who labels your data
3. **Use stratified sampling** - Balance datasets to prevent accuracy illusion
4. **Implement ensemble disagreement** - Train multiple models independently; flag samples where they disagree
5. **Continuous monitoring** - Deploy canary attacks and monitor if they are detected

---

### 6.7 Limitations and Lessons Learned

**What to discuss:**
Be honest about what didn't work and what you'd do differently.

**Topics:**

1. **CIDDS-001 Failure**
   > Our inability to achieve baseline performance on CIDDS-001 (F1≈50%) highlights the importance of dataset-specific hyperparameter tuning. The fixed 3-epoch training regime, while appropriate for CIC-IDS2017 and UNSW-NB15, proved insufficient for CIDDS-001's severe class imbalance (367:1). Future work should employ early stopping based on validation performance rather than fixed epochs, and explore class-balancing techniques (e.g., weighted loss, SMOTE).

2. **Oracle Defense Assumption**
   > Our defenses assume knowledge of the poisoning rate, representing an upper bound on defense effectiveness. Practical deployments require automatic budget estimation, which remains an open problem.

3. **Binary Classification Simplification**
   > By collapsing multi-class attack types into binary labels, we may have discarded fine-grained vulnerability patterns. Targeted attacks that flip specific attack categories (e.g., DDoS → benign) while leaving others intact could reveal different vulnerability profiles.

4. **Temporal Structure Underexplored**
   > Temporal Window poisoning was only evaluated on CUPID. Future work should explore time-series models (LSTM, Transformer) and time-aware poisoning strategies more systematically.

---

## Section 7: Conclusion

**Goal:** Concise summary of contributions, main findings, and future work.

### Structure (3-4 paragraphs)

**Paragraph 1: Restate the problem and your approach**
> Network intrusion detection systems (NIDS) rely on labeled training data, creating vulnerability to label poisoning attacks. We conducted the first systematic cross-dataset study of label poisoning on four benchmark NIDS datasets (UNSW-NB15, CIC-IDS2017, CUPID, CIDDS-001) across five model architectures and five poisoning strategies.

**Paragraph 2: Key findings**
> Our experiments revealed three critical findings: (1) Simple untargeted attacks (Class Hiding) proved more effective than sophisticated targeted strategies, challenging assumptions from adversarial ML literature. (2) Poisoned models can maintain high overall accuracy (80%+) while achieving 0% attack detection—an "accuracy illusion" that makes attacks difficult to detect in practice. (3) Dataset choice dramatically affects conclusions: CIC-IDS2017 proved highly vulnerable while UNSW-NB15 showed resilience, validating the need for cross-dataset evaluation.

**Paragraph 3: Practical implications**
> For practitioners, our findings underscore the inadequacy of accuracy as a health metric for NIDS and the critical importance of monitoring per-class recall. Standard defenses (removal, reweighting) failed at poisoning rates above 10%, indicating that current mitigation strategies are insufficient for high-intensity attacks.

**Paragraph 4: Future work**
> Future research should explore: (1) robust loss functions designed for adversarial settings (e.g., noise-robust cross-entropy), (2) active learning approaches that query suspicious labels before deployment, (3) multi-class attack-specific poisoning to reveal finer-grained vulnerabilities, and (4) temporal poisoning strategies on sequential models. Ultimately, securing NIDS training pipelines requires moving beyond outlier detection toward principled robust learning frameworks.

---

## Practical Workflow: How to Execute This Plan

### Step 1: Data Extraction (2-3 hours)
1. Open `eval_results_table.md`
2. For each dataset, create filtered views in a spreadsheet:
   - Filter by `dataset = cic`, group by `attack_type`, `poison_pct`, `defense`
   - Extract columns: `model_name`, `test_accuracy`, `macro_f1`, `pos_recall`
3. Create summary tables (like the examples above)
4. Identify key numbers for narratives (e.g., "F1 dropped from 0.97 to 0.45")

### Step 2: Write Results (5-6 hours)
1. Start with Section 5.1 (baseline table) - this is easiest
2. Write 5.2 (CIC-IDS2017) - most interesting dataset, write thoroughly
3. Write 5.3 (UNSW-NB15) - use 5.2 as template, but shorter
4. Write 5.4 (CUPID) - similar to above
5. Write 5.5 (CIDDS-001) - acknowledge failure honestly
6. Write 5.6 (overview) - synthesize patterns

**Writing style for Results:** 
- Passive voice is OK ("Performance was measured...")
- Past tense ("The model achieved...")
- No speculation ("This suggests..." belongs in Discussion)
- Lots of numbers and references to tables/figures

### Step 3: Write Discussion (4-5 hours)
1. Start with 6.1 (Accuracy Illusion) - your headline finding
2. Write 6.2 (Paradox) - second most interesting finding
3. Write 6.3-6.5 - Connect to literature, explain mechanisms
4. Write 6.6 (Implications) - Make it practical
5. Write 6.7 (Limitations) - Be honest, show maturity

**Writing style for Discussion:**
- Active voice is better ("We hypothesize...")
- Present tense for interpretations ("This indicates...")
- Speculation is encouraged ("This may occur because...")
- Connect every observation back to a "why"

### Step 4: Write Conclusion (30 minutes)
1. Use the 4-paragraph template above
2. Keep it concise (half page max)
3. End on a forward-looking note (future work)

### Step 5: Cross-Check (1 hour)
1. Verify every claim in Discussion is supported by Results
2. Verify every claim in Introduction contributions is delivered in Results/Discussion
3. Check figure references are correct
4. Ensure "Accuracy Illusion" finding is prominently featured

---

## Key Numbers You Need (Data Hunting Guide)

To write the sections, hunt for these specific numbers in `eval_results_table.md`:

### For "Accuracy Illusion" (highest priority)
Find: `dataset=cic`, `attack_type=class_hiding`, `poison_pct=020`, `defense=(empty)`
Extract: `test_accuracy`, `pos_recall`, `tn`, `fp`, `fn`, `tp` for CNN

### For "Class Hiding vs. Feature Targeting" comparison
Find: Both attack types at `poison_pct=020` on CIC-IDS2017
Extract: `macro_f1` for both
Show the contrast: "Class Hiding: 0.45, Feature: 0.95"

### For "Dataset Resilience" comparison
Find: `attack_type=class_hiding`, `poison_pct=020` for UNSW vs CIC
Extract: `macro_f1` for RF or CNN on both datasets
Show: "UNSW: 0.82, CIC: 0.45"

### For "Defense Failure" at high rates
Find: `attack_type=class_hiding`, `poison_pct=020` with `defense=removal`
Extract: `macro_f1`
Show: "Removal: 0.45 (same as no defense)"

---

## Final Checklist Before Submission

- [ ] Every figure (1-6) is referenced in Results
- [ ] "Accuracy Illusion" is demonstrated with actual numbers from your data
- [ ] CIDDS-001 failure is acknowledged honestly
- [ ] Class Hiding > Feature Targeting finding is explained
- [ ] Cross-dataset variance is discussed
- [ ] Defense limitations are characterized quantitatively
- [ ] All claims in Introduction contributions are addressed
- [ ] Conclusion mentions future work
- [ ] Limitations section shows self-awareness
- [ ] Tables have proper captions and labels

---

## Estimated Time Investment

| Section | Time | Priority |
|---------|------|----------|
| Data extraction & tables | 3 hrs | Critical |
| Section 5 (Results) | 6 hrs | Critical |
| Section 6 (Discussion) | 5 hrs | Critical |
| Section 7 (Conclusion) | 0.5 hrs | High |
| Revision & cross-check | 1.5 hrs | High |
| **Total** | **16 hrs** | |

Spread this over 2-3 days for best quality. Don't rush the Discussion—it's where you show deep understanding.

---

## Questions to Ask Yourself While Writing

**For every claim in Results:**
- Can I cite a specific table/figure?
- Is this a fact or an interpretation? (Facts → Results, Interpretations → Discussion)

**For every claim in Discussion:**
- Is this supported by data in Results?
- Have I explained WHY, not just WHAT?
- Does this connect to related work?

**For Conclusion:**
- Would a busy reviewer reading only this paragraph understand my contribution?
- Is the future work specific enough to be actionable?

---

Good luck! Follow this plan systematically and you'll produce a strong, coherent final report. 🚀
