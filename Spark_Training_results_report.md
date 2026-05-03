# Appendix Cancer Risk Prediction Model — Results Report

## 1. Project Overview

This project builds a binary classification model to predict **Appendix Cancer risk** using a large structured healthcare dataset. The target column is:

```text
Appendix_Cancer_Prediction
```

The prediction task is converted into a binary label:

| Original Target Value | Encoded Label |
|---|---:|
| No | 0.0 |
| Yes | 1.0 |

The project was implemented using **PySpark MLlib** to handle preprocessing, model training, and evaluation at scale.

---

## 2. Dataset Summary

After loading the dataset, the schema contained demographic, lifestyle, medical history, clinical measurement, and diagnosis-related columns.

### Class Distribution

| Label | Meaning | Count |
|---:|---|---:|
| 0.0 | No Appendix Cancer | 220,713 |
| 1.0 | Appendix Cancer | 39,287 |
| **Total** |  | **260,000** |

The dataset is clearly imbalanced. The negative class is much larger than the positive class.

---

## 3. Data Cleaning and Preprocessing

### Duplicate Removal

Duplicate records were removed using:

```python
df = df.dropDuplicates()
```

### Null Value Check

A null-value check was performed across all columns. Since no null values were found, no imputation or null-handling step was required.

### Dropped Leakage Columns

The following columns were removed because they could leak information that would not realistically be available before prediction:

```text
Survival_Years_After_Diagnosis
Diagnosis_Delay_Days
Treatment_Type
Tumor_Markers
```

These variables are either post-diagnosis or strongly diagnosis-dependent, so keeping them could artificially inflate model performance.

### Feature Engineering

Categorical columns were processed using:

1. `StringIndexer`
2. `OneHotEncoder`
3. `VectorAssembler`

For the GBT model, indexed categorical columns were used directly because Spark's `GBTClassifier` does not support one-hot encoded categorical vectors in the same way as Logistic Regression and Random Forest pipelines.

### Train-Test Split

The dataset was split into training and testing sets using:

```python
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)
```

The test set size was:

```text
51,840 records
```

---

## 4. Class Imbalance Handling

Two weighting strategies were used.

### Balanced Class Weights

The first weighting approach used the standard balanced-weight formula:

```python
weight_for_0 = total / (2 * majority_count)
weight_for_1 = total / (2 * minority_count)
```

This was applied to Logistic Regression, Random Forest, and GBT.

### Positive-Weighted Random Forest

A separate experiment increased the positive class weight more directly:

| Class | Training Count | Weight |
|---:|---:|---:|
| 0.0 | 176,717 | 1.0 |
| 1.0 | 31,443 | 5.6202 |

The positive class weight was calculated as:

```python
positive_weight = negative_count / positive_count
```

---

## 5. Models Trained

The following models were trained and compared:

1. Logistic Regression
2. Random Forest
3. Gradient-Boosted Tree Classifier
4. Positive-Weighted Random Forest

---


## 7. Overall Model Performance

| Model | AUC | Accuracy | F1 Score | Precision | Recall | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4969 | 0.5106 | 0.5792 | 0.7414 | 0.5106 | 135.86 sec |
| Random Forest | 0.5022 | 0.6631 | 0.6975 | 0.7439 | 0.6631 | 975.18 sec |
| GBT Classifier | 0.5004 | 0.5342 | 0.6001 | 0.7443 | 0.5342 | 1138.98 sec |
| Positive-Weighted Random Forest | 0.5022 | 0.6603 | 0.6956 | 0.7436 | 0.6603 | 1028.54 sec |

---

## 8. Confusion Matrix Results

In the following confusion matrices:

| Term | Meaning |
|---|---|
| TP | Correctly predicted Appendix Cancer cases |
| FP | Predicted Appendix Cancer, but actual class was No |
| FN | Predicted No, but actual class was Appendix Cancer |
| TN | Correctly predicted No Appendix Cancer cases |

---

### 8.1 Logistic Regression Confusion Matrix

| Actual \ Predicted | Predicted 1.0 | Predicted 0.0 | Total Actual |
|---|---:|---:|---:|
| Actual 1.0 | **3,736** | **4,108** | 7,844 |
| Actual 0.0 | **21,260** | **22,736** | 43,996 |
| Total Predicted | 24,996 | 26,844 | 51,840 |

| Metric | Value |
|---|---:|
| TP | 3,736 |
| FP | 21,260 |
| FN | 4,108 |
| TN | 22,736 |
| Positive Precision | 0.1495 |
| Positive Recall | 0.4763 |

Logistic Regression detected around **47.63%** of the actual positive cases, but its positive precision was low at **14.95%**, meaning many predicted positive cases were false positives.

---

### 8.2 Random Forest Confusion Matrix

| Actual \ Predicted | Predicted 1.0 | Predicted 0.0 | Total Actual |
|---|---:|---:|---:|
| Actual 1.0 | **2,120** | **5,724** | 7,844 |
| Actual 0.0 | **11,739** | **32,257** | 43,996 |
| Total Predicted | 13,859 | 37,981 | 51,840 |

Approximate positive-class metrics calculated from the confusion matrix:

| Metric | Value |
|---|---:|
| TP | 2,120 |
| FP | 11,739 |
| FN | 5,724 |
| TN | 32,257 |
| Positive Precision | 0.1530 |
| Positive Recall | 0.2703 |

Random Forest produced better overall accuracy and F1 than Logistic Regression, but it detected fewer actual positive cases.

---

### 8.3 GBT Classifier Confusion Matrix

| Actual \ Predicted | Predicted 1.0 | Predicted 0.0 | Total Actual |
|---|---:|---:|---:|
| Actual 1.0 | **3,583** | **4,261** | 7,844 |
| Actual 0.0 | **19,887** | **24,109** | 43,996 |
| Total Predicted | 23,470 | 28,370 | 51,840 |

Approximate positive-class metrics calculated from the confusion matrix:

| Metric | Value |
|---|---:|
| TP | 3,583 |
| FP | 19,887 |
| FN | 4,261 |
| TN | 24,109 |
| Positive Precision | 0.1527 |
| Positive Recall | 0.4568 |

GBT achieved higher positive recall than Random Forest, but its overall accuracy and F1 score were lower than Random Forest.

---

### 8.4 Positive-Weighted Random Forest Confusion Matrix

| Actual \ Predicted | Predicted 1.0 | Predicted 0.0 | Total Actual |
|---|---:|---:|---:|
| Actual 1.0 | **2,137** | **5,707** | 7,844 |
| Actual 0.0 | **11,901** | **32,095** | 43,996 |
| Total Predicted | 14,038 | 37,802 | 51,840 |

| Metric | Value |
|---|---:|
| TP | 2,137 |
| FP | 11,901 |
| FN | 5,707 |
| TN | 32,095 |
| Positive Precision | 0.1522 |
| Positive Recall | 0.2724 |

The positive-weighted Random Forest slightly increased true positives compared to the standard Random Forest, but the improvement was very small. It also slightly reduced accuracy and F1 score.

---

## 9. Best Model Selection

### Best Overall Model: Random Forest

Based on the current results, the **standard Random Forest** model is the best overall model.

| Reason | Explanation |
|---|---|
| Highest Accuracy | Random Forest achieved the highest accuracy: **0.6631** |
| Highest F1 Score | Random Forest achieved the highest F1 score: **0.6975** |
| Best balance for overall classification | It performed better than Logistic Regression, GBT, and Positive-Weighted Random Forest on overall metrics |
| Similar AUC to other models | AUC remained close to 0.50 for all models, so accuracy and F1 were more useful for comparison here |

However, this model is not ideal for detecting the positive class because its positive recall is only around **27.03%**.

---

## 10. Important Interpretation

Although Random Forest has the best overall metrics, the AUC values for all models are close to **0.50**:

| Model | AUC |
|---|---:|
| Logistic Regression | 0.4969 |
| Random Forest | 0.5022 |
| GBT Classifier | 0.5004 |
| Positive-Weighted Random Forest | 0.5022 |

An AUC close to 0.50 means the models are only slightly better than random ranking. This suggests that the remaining non-leakage features may not contain a strong signal for predicting Appendix Cancer, or that additional feature engineering and model tuning are required.

The weighted precision values appear high because the dataset is imbalanced and weighted metrics are influenced heavily by the majority class. Therefore, positive-class precision and positive-class recall are more important for understanding cancer-case detection performance.

---

## 11. Observations

1. The original dataset is highly imbalanced, with many more negative cases than positive cases.
2. Removing leakage columns caused the model performance to drop to more realistic levels.
3. Random Forest achieved the best overall accuracy and F1 score.
4. Logistic Regression achieved better positive recall than Random Forest, but it produced many false positives.
5. GBT performed similarly to Logistic Regression in terms of positive-class recall but had lower overall performance than Random Forest.
6. Increasing the positive class weight in Random Forest did not significantly improve positive recall.
7. AUC values near 0.50 indicate weak separability between the two classes.

---

## 12. Model Saving

The trained models were saved using Spark ML pipeline save methods.

```python
lr_model.write().overwrite().save("saved_models/logistic_regression_model")
rf_model_positive_weighted.write().overwrite().save("saved_models/rf_model_positive_weighted")
rf_model.write().overwrite().save("saved_models/random_forest_model")
gbt_model.write().overwrite().save("saved_models/gbt_model")
```

Saved model directories:

| Model | Saved Path |
|---|---|
| Logistic Regression | `saved_models/logistic_regression_model` |
| Random Forest | `saved_models/random_forest_model` |
| Positive-Weighted Random Forest | `saved_models/rf_model_positive_weighted` |
| GBT Classifier | `saved_models/gbt_model` |

Each saved Spark model directory contains metadata and stage folders. The complete directory should be kept together when reusing the model; it is not a single `.pkl` or `.pt` file.

---

## 13. Limitations

1. All models produced AUC values close to 0.50, suggesting weak predictive separation.
2. Positive-class precision is low for all models, meaning many positive predictions are false positives.
3. Positive-class recall is also limited, especially for Random Forest models.
4. The dataset may require better feature engineering or additional medically meaningful predictors.
5. Training time was high for tree-based models, especially GBT and Random Forest.
6. Implementing SMOTE with Spark cause OOM errors

---


## 15. Final Conclusion

The best overall model from the current experiments is the **standard Random Forest classifier**, with:

| Metric | Value |
|---|---:|
| AUC | 0.5022 |
| Accuracy | 0.6631 |
| F1 Score | 0.6975 |
| Precision | 0.7439 |
| Recall | 0.6631 |
| Training Time | 975.18 sec |

However, the model is not strong enough for reliable cancer-risk prediction because the AUC is close to 0.50 and positive-class recall is low. The current results are useful as a baseline, but more feature engineering, threshold tuning, and imbalance-handling strategies are needed before the model can be considered practically useful.
