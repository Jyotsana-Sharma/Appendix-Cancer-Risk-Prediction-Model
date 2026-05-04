# EDA & Visualization Report
## Appendix Cancer Prediction Dataset
**Author:** Jyotsana Sharma  
**Role:** EDA + Visualization + Insights  
**Date:** May 2026  
**Dataset:** `appendix_cancer_prediction_dataset.csv` — 260,000 patient records, 25 features

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Total Records | 260,000 |
| Features | 25 (demographics, lifestyle, clinical markers, treatment) |
| Target Variable | `Appendix_Cancer_Prediction` (Yes / No) |
| Missing Values | 2 columns (see Section 1b) |
| Cancer-Positive | 39,287 (15.1%) |
| Cancer-Negative | 220,713 (84.9%) |

---

## 1b. Missing Value Analysis & Handling

| Column | Missing Count | % of Total | Interpretation | Action Taken |
|--------|--------------|------------|----------------|--------------|
| `Chronic_Diseases` | 130,087 | **50.0%** | Only 2 values exist (Hypertension, Diabetes) — NaN means no chronic disease | Filled with `"None"` |
| `Treatment_Type` | 26,074 | **10.0%** | Surgery / Chemotherapy / Radiation exist — NaN means no treatment received | Filled with `"None / Untreated"` |

**Are the missing values informative?**

Cancer rate in rows where `Chronic_Diseases` is missing: **15.10%** vs present: **15.12%** — virtually identical. The missingness carries no additional signal and is safely interpretable as "no condition."

Cancer rate in rows where `Treatment_Type` is missing: **15.50%** vs present: **15.07%** — a 0.43pp difference, negligible. Missing likely means untreated at the time of record.

**Imputation strategy:** Structural imputation (NaN = a real category, not random absence). No rows were dropped.

---

The dataset covers patient demographics (age, gender, country, BMI), lifestyle factors (smoking, alcohol, physical activity, diet), clinical markers (WBC, RBC, Platelet Count, Cholesterol), and treatment outcomes.

---

## 2. Key Findings

### Finding 1 — Class Imbalance (Critical for Modeling)

The dataset is **moderately imbalanced**: 84.9% of patients are cancer-negative and only 15.1% are cancer-positive. This is approximately a 5.6:1 ratio.

**Impact:** A naive classifier that always predicts "No Cancer" would achieve 84.9% accuracy while being completely useless clinically. Any ML model built on this dataset **must** use:
- Stratified train/test splits
- Class weighting (`class_weight='balanced'`)
- Evaluation metrics beyond accuracy: AUC-ROC, F1-score (macro), Precision-Recall curve

---

### Finding 2 — Gender is the Only Statistically Significant Feature

| Gender | Cancer Rate |
|--------|------------|
| Female | 14.91% |
| Male | 15.29% |
| Other | 15.65% |

**Chi-Square Test:** χ² = 8.58, p = 0.0137 ✓ (significant at α = 0.05)

Gender is the **only feature** that passes the significance threshold. The "Other" gender category shows a marginally elevated rate. However, the absolute difference is small (~0.7 percentage points), meaning gender alone is not clinically useful for prediction.

**Demographic Bias Note:** The dataset includes a three-category gender variable (Female, Male, Other). The "Other" group shows a slightly elevated cancer rate, which warrants monitoring in any deployed model to avoid compounding discrimination.

---

### Finding 3 — Age Shows No Predictive Signal

| Group | Median Age |
|-------|-----------|
| Cancer = Yes | 53 years |
| Cancer = No | 53 years |

**Independent Samples T-Test:** t = -0.93, p = 0.352 (NOT significant)

Age is uniformly distributed between 18 and 90 across both classes. This is atypical of real-world cancer datasets where risk increases with age, and is a characteristic of the synthetic data generation process.

The cancer rate by age group fluctuates only marginally (within ±0.5% of the 15.1% baseline) — confirming age carries no predictive power here.

---

### Finding 4 — Traditional Risk Factors Show No Significant Association

| Risk Factor | Positive Group Rate | Negative Group Rate | p-value |
|-------------|--------------------|--------------------|---------|
| Smoking Status | ~15.1% | ~15.1% | 0.391 |
| Alcohol Consumption | ~15.1% | ~15.1% | 0.439 |
| Family History of Cancer | ~15.0% | ~15.2% | 0.337 |
| Genetic Mutations | ~15.1% | ~15.1% | 0.598 |
| Radiation Exposure | ~15.2% | ~15.1% | 0.326 |
| Previous Cancers | ~15.0% | ~15.1% | 0.758 |

**All chi-square tests: NOT significant (all p > 0.30)**

None of the traditional cancer risk factors — which in real epidemiological data are strongly associated with cancer incidence — show any predictive relationship with the target in this dataset. Every subgroup's cancer rate clusters within 0.3% of the global baseline.

This is the clearest indicator that this is a **synthetically generated dataset** where the cancer label was assigned with some independence from the feature values.

---

### Finding 5 — Blood & Clinical Markers Are Non-Discriminative

| Clinical Feature | t-statistic | p-value | Result |
|-----------------|------------|---------|--------|
| White Blood Cell Count | -1.001 | 0.317 | Not significant |
| Red Blood Cell Count | -0.372 | 0.710 | Not significant |
| Platelet Count | 1.503 | 0.133 | Not significant |
| Cholesterol Level | 0.906 | 0.365 | Not significant |
| Blood Pressure | 0.589 | 0.556 | Not significant |
| Diagnosis Delay (days) | -1.251 | 0.211 | Not significant |

The KDE (density) plots for each blood marker show near-perfect overlap between cancer and non-cancer patients. In a real clinical dataset, WBC count and Platelet Count would typically show measurable differences.

---

### Finding 6 — BMI Has Marginal Variation (Not Significant)

| BMI Category | Cancer Rate |
|-------------|------------|
| Underweight | 14.6% |
| Normal | 15.0% |
| Overweight | 15.3% |
| Obese I | 15.1% |
| Obese II+ | 15.2% |

The Overweight category shows the highest cancer rate at 15.3%, but the spread across all BMI categories is less than 0.7 percentage points — not statistically significant.

---

### Finding 7 — Geographic Variation Is Narrow

**Top 5 Countries by Cancer Rate:**

| Country | Cancer Rate |
|---------|------------|
| Poland | 16.5% |
| Spain | 16.3% |
| Sweden | 15.9% |
| Australia | 15.6% |
| Norway | 15.6% |

**Bottom 5 Countries by Cancer Rate:** rates range ~13–14%.

The spread between highest and lowest country rates is approximately 3–4 percentage points. No country represents a genuine outlier. Geographic features will likely contribute marginally to model performance.

---

### Finding 8 — Feature Correlation with Target (All Near Zero)

The highest point-biserial correlation between any feature and the cancer target is |r| = 0.006 (Gender). All other features have |r| < 0.003.

**Top 5 features ranked by correlation with target:**
1. Gender (|r| = 0.006, p = 0.003)
2. Platelet Count (|r| = 0.003, p = 0.133)
3. Symptom Severity (|r| = 0.003, p = 0.200)
4. Diagnosis Delay Days (|r| = 0.002, p = 0.211)
5. Alcohol Consumption (|r| = 0.002, p = 0.223)

For reference, a correlation of |r| = 0.006 explains only 0.004% of variance in the outcome — practically zero.

---

## 3. Visualizations Produced

| # | Figure | File | Key Observation |
|---|--------|------|-----------------|
| 1 | Target Class Distribution | `viz1_target_distribution.png` | 84.9% / 15.1% split |
| 2 | Age & Gender Distributions | `viz2_age_gender.png` | Uniform age; slight gender rate gap |
| 3 | Key Risk Factor Rates | `viz3_risk_factors.png` | All factors flat ~15% |
| 4 | BMI & Lifestyle | `viz4_bmi_lifestyle.png` | Minimal variation across groups |
| 5 | Correlation Heatmap | `viz5_correlation_heatmap.png` | No numeric feature correlated with target |
| 6 | Country Cancer Rates | `viz6_country_rates.png` | Narrow 13–17% band globally |
| 7 | Age Group Trends + Heatmap | `viz7_age_trends.png` | Flat trend; no age group stands out |
| 8 | Clinical Blood Markers (KDE) | `viz8_clinical_markers.png` | Overlapping distributions |
| 9 | Feature Importance Chart | `viz_feature_importance.png` | Gender tops chart at |r|=0.006 |

---

## 4. Insights for Report (Bullet Points)

- **15.1% of 260,000 patients** are cancer-positive — the dataset is class-imbalanced (5.6:1 ratio).
- **Two columns have missing values**: `Chronic_Diseases` (50% missing — imputed as "None") and `Treatment_Type` (10% missing — imputed as "None / Untreated"). Missingness is structural, not random, and carries no additional signal for the target.
- **Gender is the only statistically significant feature** (p = 0.014); all other 23 features are non-significant.
- **No blood marker, risk factor, or demographic variable** meaningfully separates cancer from non-cancer patients.
- **All feature correlations with the target are near-zero** (max |r| = 0.006), indicating the label is largely independent of the provided features.
- **The dataset exhibits characteristics of synthetic data**: uniform age distribution, risk factors uncorrelated with outcome, and identical blood marker distributions across classes.
- **Class imbalance must be addressed** in the ML pipeline using stratified splits, class weighting, or oversampling (SMOTE).
- **Geographic patterns are narrow**: Poland (16.5%) and Spain (16.3%) are the highest-rate countries, but differences are not clinically meaningful.
- **Modeling recommendation**: use tree-based ensembles (XGBoost, Random Forest) with `class_weight='balanced'`, evaluate using AUC-ROC and macro F1-score, and use SHAP to surface any latent feature interactions the univariate analysis cannot detect.

---

## 5. Storytelling Summary

> **What does the data tell us?**
>
> We analyzed 260,000 patient records across 25 variables to understand which factors predict appendix cancer. The most striking finding is that this dataset — while large — does not contain the strong clinical signals that real cancer data would. Traditional risk factors (smoking, family history, genetic mutations) and clinical blood markers (WBC, cholesterol) show no statistically significant difference between cancer and non-cancer patients.
>
> The only statistically significant signal is **gender**, with a marginal elevation in the "Other" category. This pattern, combined with the uniform age distribution and near-identical blood marker distributions, indicates the dataset was synthetically generated.
>
> For the modeling team, this means the challenge is not feature selection — it is finding **interaction effects** that individual features cannot reveal. The class imbalance (85/15) is the most actionable finding from EDA: without correcting for it, any model risks high accuracy through trivial majority-class prediction.
>
> The visualizations show clearly that no "smoking gun" feature exists in this data. The classifier must work harder — and must be evaluated more carefully — than datasets where a few features dominate the signal.

---

*EDA Notebook:* `eda_jyotsana.ipynb`  
*All visualizations saved in project directory as PNG files.*
