# Student Performance Prediction

A machine learning project analyzing factors that influence student academic performance using the UCI Student Performance dataset.

## Overview

This project compares Linear Regression and Random Forest models to predict student final grades (G3) in mathematics. Two scenarios are tested:
- **With G1 & G2**: Using previous period grades as features
- **Without G1 & G2**: Predicting performance based solely on demographic and social factors

## Project Structure
```
Programming Assignment - Kaggle Platform
├─ notebooks
│  ├─ linear_regression_test1.ipynb
│  ├─ linear_regression_test2.ipynb
│  ├─ random_forest_test1.ipynb
│  ├─ random_forest_test2.ipynb
│  └─ student_data_analysis.ipynb
├─ outputs
│  ├─ factor_importance.npy
│  ├─ metrics
│  │  └─ results.txt
│  ├─ plots
│  │  ├─ lr_without_g1g2.png
│  │  ├─ lr_with_g1g2.png
│  │  ├─ rf_without_g1g2.png
│  │  ├─ rf_without_g1g2_importance.png
│  │  ├─ rf_with_g1g2.png
│  │  └─ rf_with_g1g2_importance.png
│  └─ predictions.npy
├─ Paturusi_Ahmad_CompSci_Special_Assignment_2025.pdf
├─ README.md
├─ requirements.txt
├─ src
│  ├─ data_prep.py
│  ├─ train.py
│  └─ visualizations.py
└─ student_data
   └─ student-mat.csv

```

## Requirements

This project was developed and tested using the following environment:

- Python 3.11.14
- numpy 2.3.5
- pandas 2.3.3
- matplotlib 3.10.7
- scikit-learn 1.7.1

## Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare data and train models:**
```bash
   python src/train.py
```

2. **Generate visualizations:**
```bash
   python src/visualizations.py
```

## Key Findings

- **Random Forest with G1 & G2** achieved the best performance (R² ≈ 0.97)
- Previous grades (G1, G2) are the strongest predictors of final performance
- Without prior grades, prediction accuracy drops significantly (R² ≈ 0.17-0.23)
- Important non-grade factors include: failures, age, absences, and parental education

## Dataset

UCI Machine Learning Repository - Student Performance Dataset
- 395 student records
- 33 features including demographic, social, and academic attributes
- Target variable: G3 (final grade, 0-20)
