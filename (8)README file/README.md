# PTSD Severity Prediction in War Veterans 🧠💻

This project predicts the **severity of Post-Traumatic Stress Disorder (PTSD)** in war veterans using **machine learning algorithms**. By analyzing diverse data sources—including medical records, psychological assessments, wearable device data, and self-reported symptoms—the model enables early detection of high-risk individuals, supporting timely and personalized interventions.

## **Overview**

Veterans are particularly vulnerable to PTSD, which can lead to severe psychological, social, and physical health issues. Traditional diagnostic methods rely on self-reporting and clinical evaluations, which are often limited by subjectivity, inconsistency, and delayed detection.  

This project applies machine learning to:  
- Predict PTSD severity  
- Identify critical patterns and risk factors  
- Enable early, personalized interventions  

The approach integrates **Random Forest, CatBoost, XGBoost, Gradient Boosting**, and other algorithms to model complex interactions between psychological, demographic, and behavioral variables.

## **Features**

- **Input:** Medical records, psychological assessments, wearable data, and self-reported symptoms  
- **Output:** PTSD severity (categorical values)  
- **Algorithms Tested:**  
  - Random Forest  
  - CatBoost  
  - XGBoost  
  - Gradient Boosting  
  - AdaBoost  
  - SVM  
  - KNN  
  - ANN  
  - Naive Bayes  
  - Logistic Regression  
  - SGD  
  - Extra Trees  
  - LDA & QDA  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

## **Dataset**

The project uses a **US Veteran dataset** with comprehensive clinical and diagnostic features, including:  

- Study Class & Treatment Focus  
- Clinical Setting & Study Design  
- Symptom Clusters (Intrusion, Avoidance, Hyperarousal)  
- Comorbid conditions: Depression, Anxiety, Sleep, Anger, Substance Use  
- Quality of Life & Functioning Outcomes  
- Trauma Type & Details  
- Suicide/self-harm assessment  

This rich dataset enables precise prediction of PTSD severity.  

---

## **Data Preprocessing**

- **Categorical to Numerical Conversion:**  
  - Diagnostic Measures, Study Class, Treatment Focus, and Clinical Settings encoded using custom ranking systems.  
  - PTSD Severity encoded into High (0), Medium (1), Low (2).  

- **Normalization:**  
  Min-Max scaling applied to numeric features to bring values between 0 and 1:  

## **Feature Selection:**

 - Correlation analysis and statistical tests were performed to identify the most predictive       features.
 - Features with low correlation to PTSD severity or high multicollinearity were removed.
 - Key features retained include psychological metrics (depression, anxiety, trauma history),      demographic variables, and clinical assessment scores.

## **Feature Engineering**
 
- KMeans Clustering: Categorized PTSD severity into High, Medium, and Low. Helps identify hidden patterns and refine target labels for prediction.
- Optimized labeling provides a richer and more meaningful set of target variables for machine learning models.
- Normalized and encoded features ensure uniform scaling for model input.

## **Algorithms Used**

-Decision Tree & Random Forest: Ensemble-based tree models for robust predictions.
-Boosting Algorithms: CatBoost, XGBoost, Gradient Boosting, AdaBoost.
-Classical ML Models: SVM, KNN, Logistic Regression, Naive Bayes, LDA, QDA.
-Neural Networks: ANN for capturing complex, non-linear patterns.
-Optimization: SGD for scalable training on large datasets.

Top-performing models: Random Forest & CatBoost (98% accuracy), XGBoost & Gradient Boosting (97% accuracy).

## **Model Evaluation**

Evaluation metrics include:

-Accuracy: Correct predictions / Total predictions
-Precision: True Positives / (True Positives + False Positives)
-Recall: True Positives / (True Positives + False Negatives)
-F1-Score: Harmonic mean of Precision & Recall

Confusion matrices were computed for all models to analyze misclassifications.
Performance comparison identifies the most reliable models for deployment.


## Prerequisites
To run this project locally, ensure you have the following installed:

Python 3.7+
Streamlit
Pandas
NumPy
Scikit-learn
Joblib
Graphviz
Pydotplus

You can install these dependencies using pip:
pip install streamlit pandas numpy scikit-learn joblib graphviz pydotplus

**Setup Instructions**
1. Frontend (Streamlit)
Streamlit will handle the user interface, where users can upload a CSV file for prediction and see the results. 

To run the frontend:
Save the code provided in the app.py file.
Run the Streamlit application using the following command:
streamlit run app.py
The application will open in your default web browser, where you can interact with the project.

**Troubleshooting**:
Ensure that the Graphviz installation path is correctly set if you are using tree visualizations.
If you face any issues with libraries, ensure all dependencies are installed properly using pip.

