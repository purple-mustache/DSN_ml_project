import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pickle

# ==========================
# Streamlit App - User Interface
# ==========================
# Title of the app
st.title('Predicting Heart Disease')

# ==========================
# Sidebar - Input Sliders for User Input
# ==========================
st.header('Input Features')
# st.header('Features')
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 1, 100, 30)
    sex = st.slider('Sex', 0, 1, 0)
    cp = st.slider('Cp', 1, 4, 1)
    trestbps = st.slider('Trestbps', 70, 300, 150)
    thalach = st.slider('Thalach', 50, 350, 150)
with col2:
    exang = st.slider('Exang', 0, 1, 0)
    oldpeak = st.slider('Oldpeak', 0.0, 10.0, 5.0)
    slope = st.slider('Slope', 0.0, 5.0, 2.5)
    ca = st.slider('Ca', 0.0, 5.0, 2.5)
    thal = st.slider('Thal', 0.0, 10.0, 5.0)


# ==========================
# Prediction
# ==========================
input_data = np.array([[age, sex, cp, trestbps, thalach, exang, oldpeak, slope, ca, thal]])

loaded_model = joblib.load('svm_model.joblib')

prediction = loaded_model.predict(input_data)

# Predict the probabilities for each class using 'clf.predict_proba()'.
# This method returns the probability of the input belonging to each class.
prediction_proba = loaded_model.predict_proba(input_data)

# ==========================
# Display Prediction Results
# ==========================
# Subheader to show the prediction results.
st.subheader('Prediction')

# The '[prediction][0]' gives us the predicted species name.
st.write(f"Has Heart Disease: {True if prediction[0] == 1 else False}")

# ==========================
# Display Prediction Probabilities
# ==========================
# We also display the predicted probability for each species.
# The higher the probability, the more confident the model is that the input belongs to that species.

st.subheader('Prediction Probability')
st.write(f"Doesn't have heart disease probability: {prediction_proba[0][0]:.2f}\n\nHas heart disease probability: {prediction_proba[0][1]:.2f}")

# ref: https://medium.com/@alidu143/getting-started-with-streamlit-for-machine-learning-deployment-532e468567ce