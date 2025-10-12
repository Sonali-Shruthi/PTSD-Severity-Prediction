PTSD Severity Prediction in War Veterans 🧠💻

This project predicts the severity of Post-Traumatic Stress Disorder (PTSD) in war veterans using machine learning algorithms. By analyzing diverse data sources—including medical records, psychological assessments, wearable device data, and self-reported symptoms—the model enables early detection of high-risk individuals, supporting timely and personalized interventions.

📝 ## *Overview*

Veterans are particularly vulnerable to PTSD, which can lead to severe psychological, social, and physical health issues. Traditional diagnostic methods rely on self-reporting and clinical evaluations, which are often limited by subjectivity, inconsistency, and delayed detection.

This project applies machine learning to:

Predict PTSD severity

Identify critical patterns and risk factors

Enable early, personalized interventions

The approach integrates Random Forest, CatBoost, XGBoost, Gradient Boosting, and other algorithms to model complex interactions between psychological, demographic, and behavioral variables.





















































Prediction Of PTSD Severity in War Veterans:

Overview
This project predicts the PTSD severity of veterans using a Random Forest Regressor model. The dataset contains various psychological and demographic features such as depression, anxiety, trauma history, and other outcomes. The model is trained on a normalized dataset to predict PTSD severity. The project utilizes Streamlit for the front end and Python for the model and backend computations.

Features
Input: Various psychological and demographic features of veterans.
Output: PTSD Severity (Continuous Value).
Model: Random Forest Regressor.
Evaluation Metrics: Mean Absolute Error (MAE), R² Score, Accuracy, and Precision.
Prerequisites
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

*Setup Instructions*
1. Frontend (Streamlit)
Streamlit will handle the user interface, where users can upload a CSV file for prediction and see the results. 

To run the frontend:
Save the code provided in the app.py file.
Run the Streamlit application using the following command:
streamlit run app.py
The application will open in your default web browser, where you can interact with the project.
2. Python Model (Random Forest Regressor)
The machine learning model, trained on a dataset of veterans, is used to predict the PTSD severity based on various features. The trained model is saved as a .pkl file (e.g., random_forest_model.pkl) and loaded when the user uploads a dataset.

Model Workflow:
The model predicts the PTSD severity using the Random Forest Regressor.
Various metrics like Accuracy, Precision, Mean Absolute Error (MAE), and R² score are computed and displayed.
The first few decision trees in the random forest are visualized using Graphviz.
3. How to Use the Application
Upload your dataset:

You can upload a CSV file that contains the necessary features (excluding the PTSD Severity column).
The model will automatically predict the severity and show evaluation metrics.
Metrics Displayed:

Mean Absolute Error
R² Score
Accuracy
Precision
Decision Tree Visualization:

The first three decision trees in the trained Random Forest are displayed as PNG images.
These trees help to understand how the model makes predictions.
4. Model Evaluation and Visualizations
The model’s predictions are evaluated using the following metrics:

Mean Absolute Error (MAE): Measures the average of the absolute errors between predicted and actual values.
R² Score: Measures how well the model fits the data.
Accuracy and Precision: Metrics are calculated by converting the continuous predictions into binary values based on a threshold.
Additionally, the decision trees in the random forest are visualized to understand how individual trees contribute to the predictions.

Example Dataset:

The dataset has the following columns (excluding the PTSD Severity column for prediction):
Depression
Anxiety
Trauma History
Other psychological and demographic features , etc..

Troubleshooting
Ensure that the Graphviz installation path is correctly set if you are using tree visualizations.
If you face any issues with libraries, ensure all dependencies are installed properly using pip.
