# CodeAlpha Sales Prediction Using Python

## Project Overview

This project predicts product sales using advertising expenditure data.

The project analyzes the relationship between sales and advertising through:
- TV advertising
- Radio advertising
- Newspaper advertising

A Linear Regression model is used to predict sales.

## Dataset

The dataset used is `Advertising.csv`.

Dataset size:
- 200 rows
- 5 columns initially

After cleaning:
- 200 rows
- 4 columns

The unnecessary `Unnamed: 0` column was removed.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked dataset information.
3. Checked missing values.
4. Checked duplicate rows.
5. Removed the unnecessary `Unnamed: 0` column.
6. Selected TV, Radio, and Newspaper as features.
7. Selected Sales as the target variable.
8. Split the data into 80% training and 20% testing data.

## Machine Learning Model

Linear Regression was used for sales prediction.

Training data:
- 160 rows

Testing data:
- 40 rows

## Model Evaluation

The model produced the following results:

| Metric | Result |
|---|---:|
| MAE | 1.4608 |
| RMSE | 1.7816 |
| R² Score | 0.8994 |

The R² score is approximately 89.94%.

## Advertising Correlation Analysis

Correlation with Sales:

| Advertising Platform | Correlation |
|---|---:|
| TV | 0.7822 |
| Radio | 0.5762 |
| Newspaper | 0.2283 |

In this dataset, TV advertising has the strongest positive correlation with Sales, followed by Radio and Newspaper.

## Visualizations

### Actual vs Predicted Sales

The `actual_vs_predicted.png` graph compares actual sales values with the sales predicted by the Linear Regression model.

### Advertising Impact on Sales

The `advertising_impact.png` graph shows the correlation between each advertising platform and sales.

## Project Files

```text
Task4_Sales_Prediction/
│
├── Advertising.csv
├── sales_prediction.py
├── actual_vs_predicted.png
├── advertising_impact.png
└── README.md