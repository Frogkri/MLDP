import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model
model = joblib.load('stroke_model.pkl')

# 2. Page Configuration (Aesthetics)
st.set_page_config(page_title="StrokeGuard AI", page_icon="🧠", layout="centered")

st.title(" StrokeGuard AI: Clinical Risk Assessment")
st.markdown("""
This tool uses advanced machine learning to predict stroke risk based on patient health metrics. 
*Please fill in the patient details below for a real-time assessment.*
""")

# 3. User Input Section (User-Friendly Interface)
st.sidebar.header("Patient Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 0, 100, 50)
ever_married = st.sidebar.selectbox("Ever Married?", ["Yes", "No"])
work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])

st.sidebar.header("Medical History")
hypertension = st.sidebar.selectbox("Hypertension?", ["No", "Yes"])
heart_disease = st.sidebar.selectbox("Heart Disease?", ["No", "Yes"])
smoking = st.sidebar.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

st.sidebar.header("Clinical Measurements")
glucose = st.sidebar.number_input("Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)

# 4. Data Processing for Model (Updated to match Notebook logic)
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [float(age)], # Ensure numerical consistency
    'hypertension': [str(1 if hypertension == "Yes" else 0)], # Convert to string to match Categorical Encoder
    'heart_disease': [str(1 if heart_disease == "Yes" else 0)], # Convert to string to match Categorical Encoder
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence],
    'avg_glucose_level': [float(glucose)],
    'bmi': [float(bmi)],
    'smoking_status': [smoking]
})

# 5. Prediction Logic
if st.button("Generate Assessment"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Results")
    if prediction == 1:
        st.error(f" High Risk Detected: {probability:.2%}")
        st.write("This patient matches profiles with a high probability of stroke. Immediate clinical consultation is recommended.")
    else:
        st.success(f" Low Risk Detected: {probability:.2%}")
        st.write("The patient currently exhibits a low probability of stroke based on the provided metrics.")