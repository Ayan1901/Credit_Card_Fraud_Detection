# Credit Card Fraud Detection

An end-to-end Machine Learning pipeline to identify fraudulent credit card transactions using classification models and advanced class-imbalance techniques.

## Overview
Credit card fraud detection models face severe class imbalance since genuine transactions heavily outnumber fraudulent ones. This project cleans real-world transaction data, balances class distributions using **SMOTE**, trains evaluation models, and exports a production-ready model artifact.

## Key Performance Results

| Sampling Method | Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Imbalanced** | Logistic Regression | 99.92% | 89.06% | 62.63% | 73.54% |
| **Imbalanced** | Decision Tree | 99.88% | 64.70% | 72.52% | 68.39% |
| **Undersampling** | Logistic Regression | 94.73% | 97.91% | 92.15% | 94.94% |
| **Undersampling** | Decision Tree | 91.57% | 93.00% | 91.17% | 92.07% |
| **SMOTE (Best)** | Logistic Regression | 94.45% | 97.30% | 91.43% | 94.27% |
| **SMOTE (Best)** | **Decision Tree** | **99.82%** | **99.74%** | **99.90%** | **99.82%** |

## Features & Methodology
1. **Preprocessing**: Feature scaling (`Amount`), deduplication (removed 9,144 duplicate rows), and feature selection.
2. **Resampling Techniques**: Evaluated both Random Undersampling and Synthetic Minority Over-sampling Technique (SMOTE).
3. **Model Selection**: Trained Logistic Regression and Decision Tree Classifiers across all sampling strategies.
4. **Serialization**: Exported the top-performing Decision Tree model (`credit_card_model.pkl`) using `joblib`.

