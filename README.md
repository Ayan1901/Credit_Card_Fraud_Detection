#  Bank Customer Churn Prediction

##  Project Overview

This project uses Machine Learning to predict whether a bank customer is likely to **leave the bank (churn)** based on demographic, financial, and customer engagement characteristics.

The project compares multiple classification algorithms and evaluates them using Accuracy, Precision, Recall, and F1 Score.

---

##  Objective

The main objectives of this project are:

- Predict whether a customer will churn.
- Identify customer characteristics associated with churn.
- Compare multiple Machine Learning classification algorithms.
- Select the most suitable model for churn detection.
- Provide actionable insights for customer retention strategies.

---

##  Dataset

The dataset contains **10,000 bank customers**.

### Features

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Customer's country |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Number of years with the bank |
| Balance | Customer's account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card |
| IsActiveMember | Whether the customer is an active member |
| EstimatedSalary | Estimated customer salary |
| Exited | Target variable indicating churn |

### Target Variable

`Exited`

- `0` → Customer stayed
- `1` → Customer churned

The dataset contains:

- **7,963 customers who stayed**
- **2,037 customers who churned**
- **20.37% churn rate**

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook / Google Colab

---

##  Data Preprocessing

The following columns were removed because they are identifiers rather than useful predictive variables:

```python
["RowNumber", "CustomerId", "Surname"]
```

Missing-value analysis was performed and no missing values were found.

The categorical `Geography` variable was converted into dummy variables using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=["Geography"],
    drop_first=True
)
```

Numerical features were standardized using `StandardScaler`.

---

##  Exploratory Data Analysis

The analysis included:

- Customer churn distribution
- Age distribution
- Statistical analysis of numerical variables
- Examination of customer characteristics
- Churn class distribution

The dataset contains approximately **20.37% churned customers**, making churn detection an important business problem.

---

##  Machine Learning Models

Six classification algorithms were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes
4. Random Forest
5. XGBoost
6. Artificial Neural Network (ANN)

---

##  Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.25% | 56.00% | 21.37% | 30.94% |
| KNN | 83.10% | 61.70% | 36.90% | 46.18% |
| Naive Bayes | 82.75% | 69.67% | 21.63% | 33.01% |
| **Random Forest** | **86.80%** | **76.11%** | 47.84% | 58.75% |
| **XGBoost** | 86.35% | 70.69% | **52.16%** | **60.03%** |
| ANN | 85.80% | 70.11% | 48.35% | 57.23% |

---

##  Best Model

### Random Forest

Random Forest achieved the highest overall accuracy:

**86.80%**

### XGBoost

XGBoost achieved:

- Precision: **70.69%**
- Recall: **52.16%**
- F1 Score: **60.03%**

Since the objective is to identify customers who are likely to churn, **XGBoost is considered the stronger model for churn detection based on Recall and F1 Score**.

Recall is particularly important because failing to identify a customer who is actually going to churn may result in a lost retention opportunity.

---

##  Key Findings

### 1. Significant customer churn exists

Approximately **20.37%** of customers in the dataset have churned.

### 2. Accuracy alone is not sufficient

Random Forest achieved the highest accuracy, but XGBoost achieved better recall and F1 score.

### 3. XGBoost provides stronger churn detection

XGBoost achieved the highest recall among the tested models, making it useful for identifying potential churners.

### 4. Model comparison improves decision making

Testing multiple algorithms provides a better understanding of which approach is appropriate for the business objective.

---

##  Business Impact

A bank could use the model to:

- Identify high-risk customers.
- Prioritize retention campaigns.
- Offer personalized financial products.
- Improve customer engagement.
- Reduce unnecessary marketing expenditure.
- Potentially reduce customer churn.
- Improve customer lifetime value.

