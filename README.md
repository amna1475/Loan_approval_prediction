Loan Approval Prediction 🏦 | Machine Learning Project
📌 Overview

This project was developed as part of my internship at Elovvo
.
The system predicts whether a loan application will be approved using applicant and financial details.

🔑 Key Features

Preprocessed dataset (handled missing values, encoded categorical features)

Addressed imbalanced data using SMOTE

Implemented and compared multiple ML models:

Logistic Regression

Decision Tree

Random Forest

Evaluated using Accuracy, Precision, Recall, and F1-score

Analyzed feature dependency (dropping CIBIL score) to test robustness and avoid overfitting

⚙️ Tech Stack

Python 🐍

Pandas, NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib, Seaborn

📊 Results

Logistic Regression → ~79% accuracy

Decision Tree → ~97% accuracy (risk of overfitting with CIBIL score)

Random Forest → ~97% accuracy (better generalization)

💡 Learning Outcomes

This project helped me understand:

How preprocessing impacts ML performance

The importance of balancing datasets in classification

Why feature dependency checks are critical to avoid overfitting

How to fairly compare ML models
