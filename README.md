#  Heart Disease Prediction Using Machine Learning Algorithms

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Project Pipeline](#project-pipeline)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
5. [Data Preprocessing](#data-preprocessing)  
6. [Modeling](#modeling)  
7. [Evaluation](#evaluation)  
8. [Results](#results)  
9. [Future Work](#future-work)  
10. [Technologies Used](#technologies-used)  
11. [How to Run the Project](#how-to-run-the-project)  
12. [Contributors](#contributors)



##  Project Overview

This project aims to predict the likelihood of heart disease in patients using machine learning techniques.  
By analyzing patient data, we can assist healthcare professionals in early diagnosis and preventive treatment.  
The main goal is to build accurate classification models that can identify people at risk of heart disease.



##  Dataset

- *Source*: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
- *Number of Samples*: 303  
- *Features*: Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, ECG results, Maximum Heart Rate, Exercise-induced Angina, etc.  
- *Target Variable*: 0 (no heart disease) or 1 (presence of heart disease)



##  Project Pipeline

1. Define the problem  
2. Collect and explore the data  
3. Clean and preprocess the data  
4. Train several machine learning models  
5. Evaluate and compare their performance  
6. Suggest improvements or future directions



##  Exploratory Data Analysis (EDA)

We performed EDA to understand the structure of the data and detect patterns or anomalies:
- Visualized distributions of each feature  
- Checked correlations between features  
- Detected and handled outliers  
- Verified class imbalance in the target variable



##  Data Preprocessing

Steps we applied before training the models:
- Handled missing or incorrect values  
- Converted categorical data to numerical using encoding  
- Standardized numerical features  
- Split the dataset into training and testing sets



##  Modeling

We trained the following machine learning algorithms:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  

We used cross-validation to improve reliability and avoid overfitting.



## Evaluation

We evaluated each model using the following metrics:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- ROC-AUC Curve



##  Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 85%      | 84%       | 86%    | 85%      |
| Random Forest       | 88%      | 87%       | 89%    | 88%      |
| SVM                 | 83%      | 82%       | 85%    | 83%      |

> These values are for illustration. Replace them with your actual results.



##  Future Work

- Improve model performance through hyperparameter tuning  
- Try deep learning approaches (e.g., neural networks)  
- Deploy the model using a web framework like Flask or Streamlit  
- Collect more real-world medical data for better generalization



## Technologies Used

- Python  
- Jupyter Notebook  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn



##  How to Run the Project

1. *Clone the repository*
```bash
git clone https://github.com/yourusername/heart-disease-ml.git
cd heart-disease-ml
