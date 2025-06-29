import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error, r2_score
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydotplus

# Load the trained model
model = joblib.load('random_forest_model.pkl')  # Load your model pickle

# Set Graphviz path for rendering trees
os.environ["PATH"] += os.pathsep + r"C:/Program Files/Graphviz/bin"

def show_project_description():
    st.title('PTSD Severity Prediction Project')
    st.write("""
    ### Project Overview
    This project predicts the PTSD severity of veterans using machine learning. The dataset contains various features such as depression, anxiety, trauma history, and other psychological outcomes. 
    A **Random Forest Regressor** model is trained on a normalized dataset to predict the PTSD severity. The model has been evaluated using metrics such as Mean Absolute Error (MAE), R², accuracy, and precision.
    
    ### Key Features:
    - **Input:** Various psychological and demographic features of veterans.
    - **Output:** PTSD Severity (Continuous Value).
    - **Model:** Random Forest Regressor.

    ### How the Model Works:
    The Random Forest algorithm creates multiple decision trees during training and outputs the mean prediction of the individual trees to predict PTSD severity. It is an ensemble method, which improves accuracy and reduces overfitting.
    """)

def show_model_results(uploaded_file):
    # Load the dataset from the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Features and target
    X = df.drop('PTSD Severity', axis=1)  # Features
    y = df['PTSD Severity']  # Target (continuous)

    # Predictions using the pre-trained model
    y_pred = model.predict(X)

    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Display metrics
    st.write(f'Mean Absolute Error: {mae}')
    st.write(f'R² Score: {r2}')

    # Convert predictions to binary based on a threshold
    threshold = 0.5
    y_binary_true = (y > threshold).astype(int)  # Actual values as binary
    y_binary_pred = (y_pred > threshold).astype(int)  # Predicted values as binary

    # Calculate accuracy and precision
    accuracy = accuracy_score(y_binary_true, y_binary_pred)
    precision = precision_score(y_binary_true, y_binary_pred)

    # Display accuracy and precision
    st.write(f'Accuracy: {accuracy}')
    st.write(f'Precision: {precision}')

    # Number of trees to visualize
    num_trees_to_display = 3

    # Loop through the first few trees in the forest and display them
    for i in range(num_trees_to_display):
        # Extract the i-th tree from the random forest
        estimator = model.estimators_[i]

        # Export the tree as a dot file
        dot_data = export_graphviz(
            estimator,
            out_file=None,
            feature_names=X.columns,
            filled=True,
            rounded=True,
            special_characters=True,
            node_ids=True  # Optional: to show node IDs
        )

        # Convert to graph
        graph = pydotplus.graph_from_dot_data(dot_data)

        # Adjusting the layout to increase space between nodes
        graph.set_graph_defaults(ranksep='2.5', nodesep='1.5')  # Increase ranksep and nodesep for more space

        # Display the tree visualization
        png_data = graph.create_png()  # Get PNG data as bytes
        st.image(png_data, use_container_width=True)


        # Optionally save to a file
        graph.write_png(f"random_forest_tree_{i + 1}.png")  # Save each tree as a separate PNG file

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Project Description", "Model Prediction"])

    if page == "Project Description":
        show_project_description()
    elif page == "Model Prediction":
        st.title('Upload Your Dataset for Prediction')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            show_model_results(uploaded_file)

if __name__ == "__main__":
    main()
