# Loan Default Prediction

This repository contains a Jupyter notebook for predicting loan defaults using machine learning techniques. The notebook demonstrates data preprocessing, feature selection, model training, and evaluation steps.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

Loan default prediction is a crucial task for financial institutions to manage risk and make informed lending decisions. This project utilizes various machine learning algorithms to predict whether a loan will default based on historical data.

## Dataset

The dataset used in this project contains information on loan applications, including applicant details and loan status. The data is assumed to be in CSV format and stored in the `./Datasets/LoanData/` directory.

## Installation

To run the notebook, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- XGBoost

You can install the required packages using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan-default-predict.git
    cd loan-default-predict
    ```

2. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

3. Open `Loan Default Predict.ipynb` and run the cells to see the analysis and prediction steps.

## Models Used

The notebook explores various machine learning models, including:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Classifier

## Results

The performance of each model is evaluated using metrics such as accuracy, classification report, confusion matrix, ROC AUC score, and ROC curve.

## Contributing

Contributions are welcome! If you have any ideas or improvements, please fork the repository and submit a pull request.

---
