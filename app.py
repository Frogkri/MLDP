import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model
model = joblib.load('stroke_model.pkl')

# 2. Page Configuration (Aesthetics)
st.set_page_config(page_title="StrokeGuard AI", page_icon="🧠", layout="centered")

st.title("🧠 StrokeGuard AI: Clinical Risk Assessment")
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

# 4. Data Processing for Model
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [float(age)],
    'hypertension': [str(1 if hypertension == "Yes" else 0)],
    'heart_disease': [str(1 if heart_disease == "Yes" else 0)],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence],
    'avg_glucose_level': [float(glucose)],
    'bmi': [float(bmi)],
    'smoking_status': [smoking]
})

# 5. Prediction Logic with Enhanced Visualization
if st.button("🔍 Generate Assessment", type="primary"):
    with st.spinner("Analyzing patient data..."):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Assessment Results")
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if prediction == 1:
            # HIGH RISK DISPLAY
            st.error("### ⚠️ HIGH RISK DETECTED")
            st.markdown(f"**Stroke Probability: {probability:.1%}**")
            
            # Visual risk meter using progress bar
            st.markdown("#### Risk Level Indicator:")
            st.progress(probability, text=f"Risk Score: {probability:.1%}")
            
            # Detailed warning message
            st.warning("""
            **⚠️ Clinical Action Required:**
            
            This patient exhibits a **high probability of stroke** based on the provided health metrics. 
            
            **Recommended Actions:**
            - 🏥 Immediate clinical consultation with a neurologist or stroke specialist
            - 🩺 Comprehensive cardiovascular assessment
            - 💊 Review current medications and treatment plan
            - 📋 Implement preventive care strategies
            - 👨‍⚕️ Schedule follow-up appointments for monitoring
            """)
            
        else:
            # LOW RISK DISPLAY
            st.success("### ✅ LOW RISK DETECTED")
            st.markdown(f"**Stroke Probability: {probability:.1%}**")
            
            # Visual risk meter
            st.markdown("#### Risk Level Indicator:")
            st.progress(probability, text=f"Risk Score: {probability:.1%}")
            
            # Informational message
            st.info("""
            **Good News!**
            
            The patient currently exhibits a **low probability of stroke** based on the provided metrics.
            
            **Recommendations:**
            - ✅ Continue regular health monitoring
            - 🏃 Maintain healthy lifestyle habits
            - 📅 Schedule routine check-ups as recommended
            - 💪 Keep up with preventive care measures
            """)
    
    with col2:
        # Risk category badge
        if prediction == 1:
            st.markdown("""
            <div style='padding: 20px; background-color: #ffebee; border-radius: 10px; text-align: center;'>
                <h2 style='color: #c62828; margin: 0;'>HIGH</h2>
                <p style='color: #c62828; margin: 5px 0 0 0;'>RISK</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='padding: 20px; background-color: #e8f5e9; border-radius: 10px; text-align: center;'>
                <h2 style='color: #2e7d32; margin: 0;'>LOW</h2>
                <p style='color: #2e7d32; margin: 5px 0 0 0;'>RISK</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability meter
        st.metric(label="Stroke Probability", value=f"{probability:.1%}")
    
    # Additional risk factors section
    st.markdown("---")
    st.markdown("### 📋 Patient Risk Factors")
    
    risk_factors = []
    if hypertension == "Yes":
        risk_factors.append("🔴 Hypertension")
    if heart_disease == "Yes":
        risk_factors.append("🔴 Heart Disease")
    if smoking == "smokes":
        risk_factors.append("🔴 Current Smoker")
    if float(age) > 65:
        risk_factors.append("🟡 Age over 65")
    if float(glucose) > 140:
        risk_factors.append("🟡 Elevated Glucose")
    if float(bmi) > 30:
        risk_factors.append("🟡 Obesity (BMI > 30)")
    
    if risk_factors:
        st.warning("**Identified Risk Factors:**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    else:
        st.success("✅ No major risk factors identified")

# Footer
st.markdown("---")
st.caption("⚕️ This tool is for clinical decision support only. Always consult with healthcare professionals for diagnosis and treatment.")