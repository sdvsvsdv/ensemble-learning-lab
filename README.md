# EnsembleLab: Ensembles versus Gradient Boosting on the UCI Adult Census Income Dataset

An end-to-end, leakage-safe machine learning study that benchmarks three advanced ensemble paradigms (Soft Voting, Blending, Stacking) against three modern gradient boosting frameworks (XGBoost, LightGBM, CatBoost) for binary income classification. Every model is tuned with Optuna Bayesian optimization against the positive-class F1-score, calibrated with a post-hoc decision threshold, and evaluated once on an untouched hold-out set. The study closes with an interpretability audit comparing native tree gain with test-set permutation importance.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Dataset](#2-dataset)
3. [Preprocessing and Feature Engineering](#3-preprocessing-and-feature-engineering)
4. [Architectural Strategy: Two-Track Benchmarking Framework](#4-architectural-strategy-two-track-benchmarking-framework)
5. [Bayesian Optimization Methodology and Results](#5-bayesian-optimization-methodology-and-results)
6. [Final Benchmark on the Hold-out Test Set](#6-final-benchmark-on-the-hold-out-test-set)
7. [Model Interpretability and Feature Attribution Audit](#7-model-interpretability-and-feature-attribution-audit)
8. [Key Findings and Conclusions](#8-key-findings-and-conclusions)
9. [Repository Structure](#9-repository-structure)
10. [Installation and Execution](#10-installation-and-execution)


---

## 1. Problem Definition

**Objective.** Predict whether an individual's annual income exceeds USD 50,000 from demographic and employment attributes recorded in the 1994 U.S. Census Bureau database.

**Formulation.** Binary supervised classification, where the target is encoded as

$$
y_i = \begin{cases} 1 & \text{if income} > 50\text{K} \\ 0 & \text{if income} \leq 50\text{K} \end{cases}
$$

**Primary metric.** The F1-score of the positive class (`>50K`):

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}
$$

Accuracy is deliberately not used for model selection. With roughly three quarters of the records in the majority class, a trivial classifier that always predicts `<=50K` would attain about 76% accuracy while having an F1-score of zero on the class of interest. F1 on the minority class penalizes both false positives and false negatives and therefore reflects the practical trade-off. ROC-AUC, PR-AUC, precision, recall, the absolute false positive count and inference latency are reported as secondary metrics.

---

## 2. Dataset

| Property | Value |
| :--- | :--- |
| Source | UCI Machine Learning Repository, Adult Census Income |
| Records | 32,561 |
| Feature types | Mixed continuous and categorical |
| Class `<=50K` (label 0) | approximately 75.92% |
| Class `>50K` (label 1) | approximately 24.08% |
| Imbalance ratio | approximately 3.15 : 1 |

The imbalance is moderate rather than extreme. It is large enough to bias an unweighted learner towards the majority class and to make the default 0.50 decision threshold suboptimal, but not so severe that resampling methods such as SMOTE are required. This project instead addresses the imbalance through (a) stratified partitioning, (b) tempered class weighting tuned by Optuna, and (c) post-hoc threshold calibration.

---

## 3. Preprocessing and Feature Engineering

### 3.1 Data Partitioning and Leakage Prevention

A stratified 80/20 train-test split (`random_state=42`, `stratify=y`) is performed **before** any statistic is computed from the data. Stratification preserves the 75.92 / 24.08 class ratio in both partitions. Every learned quantity (imputation modes, scaler parameters, encoder vocabularies, and model weights) is fitted on `X_train` only and merely applied to `X_test`. This guarantees zero train-to-test information leakage, so the reported hold-out metrics are unbiased estimates of generalization performance.

### 3.2 Data Cleaning

| Step | Treatment | Rationale |
| :--- | :--- | :--- |
| Missing value encoding | Entries originally encoded as `?` are converted to `NaN` | Makes missingness explicit and machine-detectable |
| Categorical imputation | `workclass`, `occupation` and `native-country` are imputed with the statistical mode computed exclusively on `X_train` and broadcast to `X_test` | Preserves all records while preventing test statistics from influencing training |
| Drop `education` | Removed | Perfectly collinear with `education_num` (ordinal scale 1 to 16); the ordinal encoding retains the rank information in a single compact feature |
| Drop `fnlwgt` | Removed | The final sampling weight is a survey sampling artifact (the number of people in the population a record represents), not a predictive human attribute |

### 3.3 Feature Engineering

| Engineered Feature | Definition | Motivation |
| :--- | :--- | :--- |
| `capital_net` | `capital-gain` - `capital-loss` | Net financial return of an individual |
| `has_capital_gain` | Indicator of `capital-gain > 0` | Both capital variables are more than 90% zeros; the binary flag isolates the presence of asset income |
| `has_capital_loss` | Indicator of `capital-loss > 0` | Same reasoning for losses |
| `is_native_us` | `native-country == United-States` vs. other | Consolidates a high-cardinality categorical dominated by a single level, reducing sparsity after one-hot encoding |
| `is_married` | 1 for `Married-civ-spouse` and `Married-AF-spouse`, 0 otherwise | Collapses `marital-status` into the economically meaningful distinction (dual-income households) |
| `is_overtime` | Indicator of `hours-per-week > 40` | Encodes workload commitment as an explicit threshold effect |

### 3.4 Non-Linear Log Transformations

The variables `capital-gain` and `capital-loss` are extremely right-skewed: over 90% of values are zero and the maximum reaches 99,999. They are transformed with

$$
x' = \ln(1 + x)
$$

via `np.log1p`, which is well-defined at zero and compresses the heavy tail so that distance-based and linear learners (KNN, Logistic Regression) are not dominated by a handful of extreme values.

Design decisions on what is **not** transformed:

- `age`, `education_num` and `hours_per_week` are bounded or ordinal and remain in linear space.
- `capital_net` is excluded from the logarithmic transform because it can be negative, for which the logarithm is undefined.
- Tree-based models are invariant to monotonic transformations, so the transform primarily benefits the linear and distance-based components of the ensembles.

### 3.5 Sanitization and Column Grouping

Feature names are normalized by replacing `-` with `_` (for example `capital-gain` becomes `capital_gain`). Several libraries (notably LightGBM and XGBoost) restrict special characters in feature names, so this step guarantees cross-library serialization compatibility. The columns are then explicitly separated into `cat_cols` and `num_cols`, which drive the `ColumnTransformer` definitions of every pipeline.

---

## 4. Architectural Strategy: Two-Track Benchmarking Framework

All models are tuned with **Optuna** (Bayesian optimization using a Tree-structured Parzen Estimator sampler) with the positive-class **F1-score** as the objective. The design pits two philosophies against each other.

### Track A: Advanced Ensemble Paradigms (The Ensembles Battle)

**A1. Soft Voting Classifier.** A heterogeneous combination of Logistic Regression, Gradient Boosting Classifier (GBM) and K-Nearest Neighbors. The final probability is the weighted average

$$
\hat{p}(y=1 \mid x) = \frac{\sum_{m} w_m \, \hat{p}_m(y=1 \mid x)}{\sum_{m} w_m}
$$

Preprocessing is isolated per model inside a `ColumnTransformer`: `RobustScaler` for the scale-sensitive Logistic Regression and KNN, a scale-invariant passthrough for the GBM, and `OneHotEncoder` for categorical variables. Optuna tunes the base-model hyperparameters **and** the voting weights concurrently, so the weights are optimized jointly with the models they combine rather than fixed heuristically.

**A2. Blending Ensemble.** Base models (Random Forest, KNN, Logistic Regression) are trained on an 80% base split of the training data. Their predicted probabilities on the remaining 20% hold-out blend set form the meta-feature matrix used to train an L2-regularized Logistic Regression meta-learner. Blending is simple and leak-free, but the meta-learner sees only 20% of the training data.

**A3. Stacking Classifier (Out-Of-Fold Meta-Learner).** The same base estimators (Random Forest, KNN, Logistic Regression) and the same meta-learner (Logistic Regression) are used. Instead of a single hold-out, the meta-features are Out-Of-Fold (OOF) probabilities produced by 3-fold internal Stratified Cross-Validation. Each training record is therefore scored by base models that never saw it, so the meta-learner is trained on the full dataset without sacrificing data. A hold-out validation scheme is used for the Optuna objective in order to avoid the computational cost of nested cross-validation.

### Track B: Gradient Boosting Frameworks (The Boosting Battle)

| Framework | Tree growth strategy | Categorical handling | Imbalance handling | Distinguishing mechanics |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Level-wise (`grow_policy='depthwise'`) | `OneHotEncoder` | `scale_pos_weight` tuned | Histogram quantization (`tree_method='hist'`) |
| **LightGBM** | Leaf-wise (best-first), constrained by `num_leaves` and `max_depth` | Native, via histogram sorting of categories (`categorical_feature`) | `scale_pos_weight` tuned | `bagging_fraction` and `feature_fraction` for stochastic regularization; leaf-count constraint (`num_leaves: 26`, `max_depth: 7`) |
| **CatBoost** | Symmetric oblivious trees (`grow_policy='SymmetricTree'`) | Native ordered target statistics with on-the-fly categorical feature combinations | `auto_class_weights='SqrtBalanced'` | Bayesian bootstrap; ordered boosting reduces target leakage and prediction shift |

The comparison isolates structural differences: level-wise growth yields balanced trees and predictable cost; leaf-wise growth reduces loss faster per split but can overfit without a leaf cap; symmetric trees act as a strong regularizer and give very fast inference.

---

## 5. Bayesian Optimization Methodology and Results

### 5.1 Procedure

- Objective: maximize cross-validated (or hold-out, for Stacking) F1 on the positive class using the training partition only.
- The test partition is never touched during tuning, threshold selection of the final comparison, or feature selection.
- Tuning wall-clock time is recorded for the boosting frameworks as a practical cost metric.

### 5.2 Optimal Hyperparameters: Track A

| Model | Optimal Hyperparameters | Optuna F1 |
| :--- | :--- | :---: |
| **Voting** | `gbm_n_estimators: 350`, `gbm_max_depth: 5`, `gbm_learning_rate: 0.0332`, `knn_n_neighbors: 17`, `lr_C: 0.1924`, weights `w_gbm: 1.05`, `w_lr: 0.87`, `w_knn: 0.86` | 0.7255 (CV) |
| **Blending** | `rf_n_estimators: 150`, `rf_max_depth: 8`, `knn_n_neighbors: 21`, `meta_C: 28.26` | 0.7147 (CV) |
| **Stacking** | `rf_n_estimators: 120`, `rf_max_depth: 16`, `knn_n_neighbors: 19`, `meta_C: 3.87` | 0.7050 (CV) |

### 5.3 Optimal Hyperparameters: Track B

| Model | Optimal Hyperparameters | Tuning time | Optuna F1 |
| :--- | :--- | :---: | :---: |
| **XGBoost** | `max_depth: 7`, `min_child_weight: 9`, `learning_rate: 0.1367`, `n_estimators: 250`, `subsample: 0.79`, `colsample_bytree: 0.76`, `scale_pos_weight: 1.3026` | approx. 60 s (fast) | 0.7189 |
| **LightGBM** | `max_depth: 7`, `num_leaves: 26`, `min_child_samples: 22`, `learning_rate: 0.0475`, `n_estimators: 450`, `bagging_fraction: 0.794`, `feature_fraction: 0.685`, `reg_lambda: 0.388`, `cat_l2: 19.62`, `scale_pos_weight: 1.2610` | approx. 35 s (ultra-fast) | 0.7227 |
| **CatBoost** | `depth: 5`, `l2_leaf_reg: 6.31`, `border_count: 32`, `learning_rate: 0.1549`, `iterations: 1500`, `bagging_temperature: 0.554`, `rsm: 0.625`, `auto_class_weights='SqrtBalanced'` | approx. 360 s (thorough) | 0.7233 |

### 5.4 Key Hyperparameter Insight: Tempered Class Penalties

With a 3.15 : 1 imbalance, the theoretically "balanced" weight for the positive class is 3.15. Optuna consistently rejected this value:

| Framework | Optimal class penalty | Theoretical balanced value |
| :--- | :---: | :---: |
| XGBoost | `scale_pos_weight` = 1.3026 | 3.15 |
| LightGBM | `scale_pos_weight` = 1.2610 | 3.15 |
| CatBoost | `SqrtBalanced` (approximately $\sqrt{3.15} \approx 1.77$) | 3.15 |

Full re-balancing over-corrects: it pushes the decision boundary deep into the majority class, inflating recall at the expense of precision and producing many false positives. Because F1 weighs precision and recall symmetrically, the optimum lies at a partial correction (approximately 1.26 to 1.30 for the GBDTs and the square-root rule for CatBoost). This retains high precision while still compensating for the majority-class bias. The post-hoc threshold calibration in Section 6.3 then fine-tunes the remaining trade-off.

---

## 6. Final Benchmark on the Hold-out Test Set

All figures below are computed once on the untouched `X_test` and `y_test`. "Tuned F1" is obtained at the optimal decision threshold (see Section 6.3). Latency is the mean inference time per sample.

| Model | Track | Default F1 (0.50) | Optimal Threshold | Tuned F1-Score | Precision (>50K) | Recall (>50K) | ROC-AUC | PR-AUC | False Positives (FP) | Latency (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Champion)** | Track B (Boosting) | 0.7334 | **0.46** | **0.7405** | **0.7173** | 0.7653 | **0.9311** | **0.8351** | **473** | **0.0341** |
| **XGBoost** | Track B (Boosting) | **0.7361** | 0.40 | 0.7375 | 0.6751 | **0.8125** | 0.9308 | 0.8345 | 613 | 0.0121 |
| **CatBoost** | Track B (Boosting) | 0.7326 | 0.46 | 0.7356 | 0.6899 | 0.7876 | 0.9268 | 0.8240 | 555 | 0.0125 |
| **Voting Classifier** | Track A (Ensemble) | 0.7314 | 0.48 | 0.7329 | 0.7120 | 0.7551 | 0.9265 | 0.8297 | 479 | 3.1956 |
| **Blending Ensemble** | Track A (Ensemble) | 0.7252 | 0.38 | 0.7309 | 0.6942 | 0.7717 | 0.9224 | 0.8207 | 533 | 1.7036 |
| **Stacking Classifier** | Track A (Ensemble) | 0.7139 | 0.70 | 0.7299 | 0.7096 | 0.7513 | 0.9246 | 0.8231 | 482 | 1.8565 |

### 6.1 Reading the Table

- **LightGBM is the overall champion.** It attains the highest tuned F1 (0.7405), the highest precision (0.7173), the best ROC-AUC (0.9311) and PR-AUC (0.8351), and the fewest false positives (473) of all six models.
- **XGBoost** has the highest default-threshold F1 (0.7361) and the highest recall (0.8125), but it achieves this by accepting the most false positives (613). Its optimal threshold of 0.40 shows a model that is best operated in a high-recall regime, and its tuned F1 (0.7375) is only marginally above the default, so threshold calibration contributes little here.
- **CatBoost** is a close third on tuned F1 (0.7356) with strong recall (0.7876) and very low latency, at the cost of the longest tuning time (approximately 360 s).
- **Voting** is the strongest ensemble and approaches the boosters on precision (0.7120) and false positives (479), but trails on every threshold-independent metric.
- **Blending and Stacking** are the weakest on F1 and AUC. Stacking's default F1 (0.7139) is the lowest of all models, and its optimal threshold of 0.70 indicates that its meta-learner outputs are poorly calibrated at 0.50; thresholding recovers most, but not all, of the gap.

The spread in tuned F1 across all six models is only about 0.011 (0.7299 to 0.7405). The ranking is therefore informative about efficiency and robustness (precision, false positives, latency) more than about large differences in raw discriminative power. Differences of this magnitude on a test set of 6,513 records should be interpreted with appropriate caution regarding sampling variance.

### 6.2 The Ensemble Paradox

The central empirical finding is that sophisticated multi-model pipelines **did not outperform** individual, well-tuned gradient boosted decision trees, while carrying a large operational penalty.

**Accuracy.** The best ensemble (Voting, tuned F1 0.7329) trails the best single model (LightGBM, 0.7405). On ROC-AUC the gap is 0.9265 versus 0.9311, and on PR-AUC 0.8297 versus 0.8351.

**Latency.** Relative to LightGBM at 0.0341 ms/sample, the ensembles are slower by:

| Ensemble | Latency (ms/sample) | Slowdown vs. LightGBM |
| :--- | :---: | :---: |
| Blending | 1.7036 | approx. 50x |
| Stacking | 1.8565 | approx. 54x |
| Voting | 3.1956 | approx. 94x |

Against the fastest booster, XGBoost (0.0121 ms/sample), the penalties are larger still, roughly 140x to 260x.

**Why ensembling underdelivered.** Several mechanisms explain the paradox:

1. **The base learners are not complementary enough.** Ensembling gains depend on diverse, uncorrelated errors. Here the strongest members (GBM, Random Forest) are tree-based and tend to err on the same borderline records, while the weaker members (KNN, Logistic Regression) add diversity but also add bias. A boosted tree model already represents the nonlinear interactions that KNN and LR contribute.
2. **Boosting is itself an ensemble.** LightGBM, XGBoost and CatBoost each combine hundreds to over a thousand trees, fitted sequentially to correct residual errors. Wrapping a second ensembling layer around weak or redundant members yields diminishing returns.
3. **The dataset is small and low-dimensional.** With roughly 26,000 training rows and a modest number of features, the achievable signal is largely saturated by a single strong learner; there is little headroom for a meta-learner to exploit. This is also consistent with the Bayes error floor imposed by unobserved determinants of income.
4. **Meta-learners add estimation variance.** Blending trains its meta-learner on only 20% of the data, and Stacking relies on 3-fold OOF predictions that are noisy approximations of true out-of-sample behavior. Both can overfit the meta-features, which is visible in their lower AUC.
5. **Weaker base learners in the pool.** The Track A base pool for Blending and Stacking (Random Forest, KNN, Logistic Regression) contains no boosted model, so its ceiling is structurally lower than that of the Track B champions.

**Engineering consequence.** A 50x to 94x increase in inference cost, combined with higher maintenance complexity (multiple fitted models, multiple preprocessing branches, a meta-learner to version and monitor), buys no improvement. For this problem, a single regularized gradient boosted model is the Pareto-optimal choice across accuracy, precision and latency.

### 6.3 Impact of Threshold Calibration

A classifier outputs a calibrated-ish probability $\hat{p}$; the hard label is $\hat{y} = \mathbb{1}[\hat{p} \geq \tau]$. The default $\tau = 0.50$ is optimal only for symmetric costs and balanced classes. With a 24% positive prevalence, and with class weights already altering the probability scale, the F1-optimal $\tau$ is generally different from 0.50.

The optimal threshold is found by sweeping $\tau$ and maximizing F1:

$$
\tau^{*} = \arg\max_{\tau \in [0,1]} F_1(\tau)
$$

Results for the champion:

| LightGBM | Threshold | F1 |
| :--- | :---: | :---: |
| Default | 0.50 | 0.7334 |
| Calibrated | 0.46 | 0.7405 |
| Improvement | | +0.0071 |

Moving the threshold from 0.50 to 0.46 raised LightGBM's F1 by 0.0071 **without any retraining**, by shifting the operating point towards slightly higher recall at a small cost in precision. The effect varies by model:

- **Stacking** benefits the most (0.7139 to 0.7299, +0.0160) at an extreme threshold of 0.70, which exposes the miscalibration of its meta-learner.
- **Blending** gains +0.0057 at a threshold of 0.38.
- **Voting, CatBoost** gain only +0.0015 and +0.0030 respectively, as they are already close to well calibrated at their optimal point.
- **XGBoost** gains just +0.0014, because its `scale_pos_weight` already shifts its operating point.

Threshold calibration is a zero-cost deployment lever: it requires only a validation sweep and leaves the fitted model untouched. In a production setting the threshold should be selected on a validation partition (or via cross-validated predictions on the training set) and then frozen, so that the test estimate remains unbiased. It can also be re-tuned if the business cost ratio between false positives and false negatives changes.

---

## 7. Model Interpretability and Feature Attribution Audit

Two complementary attribution methods are used on the champion model:

1. **Native tree gain.** The total reduction in loss contributed by splits on each feature during training. It is cheap but computed on training data and biased towards high-cardinality or easily splittable features.
2. **Test-set permutation importance.** The drop in F1 on `X_test` when a single feature's values are randomly shuffled, breaking its association with the target:

$$
\text{PI}_j = F_1(\text{baseline}) - \mathbb{E}_{\pi}\!\left[ F_1\!\left(X^{(\pi, j)}\right) \right]
$$

It measures what the model actually relies on to generalize to unseen data and is model-agnostic.

### 7.1 Top Five Drivers by Test-Set Permutation Importance

| Rank | Feature | F1 drop when shuffled | Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `education_num` | -0.0731 | The primary generalization anchor; educational attainment strongly gates earning potential |
| 2 | `age` | -0.0652 | Proxy for career accumulation, experience and seniority |
| 3 | `occupation` | -0.0491 | Industry and role compensation ceilings |
| 4 | `is_married` | -0.0432 | Socio-economic stability and the dual-income household effect |
| 5 | `capital_gain` | -0.0427 | Direct signal of wealth and asset ownership |

### 7.2 Gain versus Permutation Discrepancy

The two methods disagree in an instructive way. Native tree gain favored high-level structural features that the model splits on early: `relationship` (18.5%) and `is_married` (13.7%). Permutation importance on unseen data instead ranks the continuous human-capital variables `education_num` and `age` at the top.

The reconciliation is that `relationship` and `is_married` are highly correlated, coarse, low-cardinality summaries of household status. The model uses them for large initial splits, which earns a large share of gain. But much of their information is redundant with each other and with `age`, so when one is permuted the model can partially recover the signal from correlated features, limiting the F1 loss. Conversely, `education_num` and `age` are fine-grained and carry information that no other feature substitutes, so removing them damages generalization more. Permutation importance on held-out data is therefore the more reliable guide to what drives out-of-sample performance, and gain should be treated as a description of how the trees were built, not of what matters for prediction.

### 7.3 Feature Engineering Validation

The engineered binary `is_married` contributed **13.7%** of total gain, versus only **3.6%** for the raw multi-class `marital_status`. This supports the domain-driven aggregation: collapsing seven marital categories into a single economically meaningful flag concentrated the signal into a feature the trees could exploit with one split, rather than diluting it across many sparse categories. `is_married` also ranks fourth under permutation importance, confirming that the benefit extends to held-out data and is not a training artifact.

---

## 8. Key Findings and Conclusions

1. **A tuned LightGBM is the champion.** It achieves tuned F1 0.7405, ROC-AUC 0.9311, PR-AUC 0.8351 and the lowest false positive count (473), with ultra-fast tuning (approximately 35 s) and sub-0.04 ms inference.
2. **The Ensemble Paradox holds.** Voting, Blending and Stacking underperform the best boosters on F1 and AUC while costing 50x to 94x more per prediction than LightGBM.
3. **Tempered class weighting beats full re-balancing.** Optuna selected penalties of about 1.26 to 1.30 (and `SqrtBalanced`) rather than the theoretical 3.15, protecting precision.
4. **Threshold calibration is a free gain.** LightGBM improved from 0.7334 to 0.7405 F1 by moving the threshold to 0.46, with no retraining.
5. **Leakage-safe design matters.** Splitting before imputation, scaling and encoding keeps all test metrics unbiased.
6. **Interpretability supports domain logic.** Education, age, occupation, marital status and capital gains are the dominant, intuitive drivers, and the engineered `is_married` feature is validated by both gain and permutation analyses.
7. **Tuning cost differs substantially** across boosters (LightGBM approx. 35 s, XGBoost approx. 60 s, CatBoost approx. 360 s) for near-identical accuracy, which is a practical selection criterion.

---

## 9. Repository Structure

```text
EnsembleLab/
|-- README.md                      # Project documentation (this file)
|-- requirements.txt               # Pinned Python dependencies
|-- LICENSE                        # Repository license
|-- .gitignore
|-- notebooks/
|   `-- EnsembleLab.ipynb          # End-to-end notebook: EDA, preprocessing, tuning, evaluation, interpretability
|-- data/
|   `-- adult.data                 # Optional local copy of the UCI file (downloaded automatically if absent)
|-- figures/                       # Exported plots (confusion matrices, PR/ROC curves, importance charts)
`-- results/                       # Optional exported metrics tables and Optuna study summaries
```

The notebook is self-contained: it downloads the dataset directly from the UCI repository, so the `data/`, `figures/` and `results/` directories are optional conveniences for local caching and exported artifacts.

---

## 10. Installation and Execution

### 10.1 Requirements

- Python 3.9 or newer
- Approximately 2 GB of free memory and any modern CPU (no GPU is required)
- Internet access on the first run to download the dataset

Core dependencies:

| Library | Purpose |
| :--- | :--- |
| `numpy`, `pandas` | Numerical computing and data manipulation |
| `scikit-learn` | Pipelines, preprocessing, Voting, Stacking, Random Forest, KNN, Logistic Regression, metrics, permutation importance |
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `catboost` | CatBoost classifier |
| `optuna` | Bayesian hyperparameter optimization |
| `matplotlib`, `seaborn` | Visualization |
| `jupyter` | Notebook execution |

An example `requirements.txt`:

```text
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
optuna
matplotlib
seaborn
jupyter
```

For exact reproducibility, pin the versions from your environment with `pip freeze > requirements.txt`.

### 10.2 Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/EnsembleLab.git
cd EnsembleLab

# 2. Create and activate a virtual environment
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 10.3 Running the Project

```bash
# Option A: interactive
jupyter notebook notebooks/EnsembleLab.ipynb

# Option B: headless execution, saving the executed notebook
jupyter nbconvert --to notebook --execute notebooks/EnsembleLab.ipynb \
    --output EnsembleLab_executed.ipynb
```

Expected behavior:

- The full run, including all six Optuna studies, completes in a matter of minutes on a standard CPU; CatBoost dominates the runtime (approximately 360 s of tuning).
- Fixed seeds (`random_state=42`) are used for the split and the models. Small numerical differences can still arise across library versions and operating systems, and multi-threaded tree construction may introduce minor non-determinism.

---

