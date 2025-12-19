# Student Performance Prediction

A machine learning project analyzing factors that influence student academic performance using the UCI Student Performance dataset.

## Overview

This project compares Linear Regression and Random Forest models to predict student final grades (G3) in mathematics. Two scenarios are tested:
- **With G1 & G2**: Using previous period grades as features
- **Without G1 & G2**: Predicting performance based solely on demographic and social factors

## Project Structure
```
├── notebooks/               
│   ├── linear_regression_test1.ipynb
│   ├── linear_regression_test2.ipynb
│   ├── random_forest_test1.ipynb
│   ├── random_forest_test2.ipynb
│   ├── student_data_analysis.ipynb  #exploring the dataset 
├── src/
│   ├── data_prep.py          # Data loading and preprocessing
│   ├── train.py              # Model training and evaluation
│   └── visualizations.py     # Results visualization
├── student_data/
│   └── student-mat.csv       # Dataset
└── outputs/
   ├── metrics/
   │   └── results.txt       # Model comparison results
   └── plots/                # Visualization outputs
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

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
