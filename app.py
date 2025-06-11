import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

st.title("Iris Flower Classifier")

# Input features
sepal_length = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.number_input("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.number_input("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.number_input("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    class_names = ["Setosa", "Versicolor", "Virginica"]
    st.write("Prediction:", class_names[prediction])
