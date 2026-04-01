# CombatCare: Prediction of PTSD Severity in War Veterans

This project predicts the severity of Post-Traumatic Stress Disorder (PTSD) in war veterans using machine learning models. By analyzing medical records, psychological assessments, wearable data, and self-reported symptoms, the system enables early identification of high-risk individuals and supports timely intervention.

## Overview

PTSD is a critical mental health issue among veterans, often leading to severe psychological and social challenges. Traditional diagnosis relies on self-reporting and clinical evaluation, which may be subjective and delayed.

This project applies machine learning to:
- Predict PTSD severity
- Identify key risk factors and patterns
- Support early and personalized intervention strategies

The system models complex relationships between psychological, demographic, and behavioral variables using multiple machine learning approaches.

## Features

- Input: Medical records, psychological assessments, wearable data, and self-reported symptoms  
- Output: PTSD severity classification (High, Medium, Low)  

### Algorithms Used
- Random Forest  
- CatBoost  
- XGBoost  
- Gradient Boosting  
- AdaBoost  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Artificial Neural Networks (ANN)  
- Naive Bayes  
- Logistic Regression  
- Stochastic Gradient Descent (SGD)  
- Extra Trees  
- Linear Discriminant Analysis (LDA)  
- Quadratic Discriminant Analysis (QDA)  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

## Dataset

The project uses a US veteran dataset with detailed clinical and diagnostic features, including:

- Study class and treatment focus  
- Clinical setting and study design  
- Symptom clusters such as intrusion, avoidance, and hyperarousal  
- Comorbid conditions including depression, anxiety, sleep disorders, anger, and substance use  
- Quality of life and functional outcomes  
- Trauma type and exposure details  
- Suicide and self-harm assessment  

## Data Preprocessing

- Converted categorical variables into numerical representations using custom encoding  
- Encoded PTSD severity as High (0), Medium (1), Low (2)  
- Applied Min-Max normalization to scale features between 0 and 1  

## Feature Selection

- Performed correlation analysis and statistical testing to identify relevant features  
- Removed features with low predictive value or high multicollinearity  
- Retained key psychological, demographic, and clinical variables  

## Feature Engineering

- Applied KMeans clustering to refine PTSD severity categorization  
- Generated improved target labels for better model performance  
- Ensured consistent scaling and encoding across all features  

## Model Development

- Built and compared multiple machine learning models  
- Tuned hyperparameters for optimal performance  
- Evaluated models using classification metrics and confusion matrices  

### Top Performing Models
- Random Forest and CatBoost achieved up to 98% accuracy  
- XGBoost and Gradient Boosting achieved up to 97% accuracy  

## Results

The models demonstrate strong predictive performance and the ability to capture complex patterns in clinical data. The system can assist in identifying high-risk individuals and support data-driven decision making in mental health care.

## Conclusion

CombatCare provides a scalable and effective approach for predicting PTSD severity using machine learning. The system highlights the potential of AI-driven healthcare solutions in improving early diagnosis and intervention for vulnerable populations.
