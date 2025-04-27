import streamlit as st
import pickle
import numpy as np

st.title("🏡 Home Loan Approval Prediction")
st.write("Enter details below to check if your loan will be approved.")

# Input fields
loan_id = st.text_input("Loan ID")
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
annual_income = st.number_input("Annual Income (in $)", min_value=0.0, step=1000.0)
loan_amount = st.number_input("Loan Amount (in $)", min_value=0.0, step=1000.0)
loan_term = st.number_input("Loan Term (in months)", min_value=1, max_value=360, step=1)
cibil_score = st.number_input("CIVIL Score", min_value=300, max_value=900, step=1)
residential_assets = st.number_input("Residential Assets Value (in $)", min_value=0.0, step=1000.0)
commercial_assets = st.number_input("Commercial Assets Value (in $)", min_value=0.0, step=1000.0)
luxury_assets = st.number_input("Luxury Assets Value (in $)", min_value=0.0, step=1000.0)
bank_assets = st.number_input("Bank Asset Value (in $)", min_value=0.0, step=1000.0)

# Additional missing feature
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
credit_history = 1 if credit_history == "Good" else 0

# Load model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

if st.button("Check Approval"):
    # Convert categorical inputs to numerical
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    features = np.array([no_of_dependents, education, self_employed, annual_income, loan_amount, loan_term,
                         cibil_score, residential_assets, commercial_assets, luxury_assets, bank_assets,
                         credit_history])

    if features.shape[0] != 12:
        st.error(f"Feature shape mismatch: Expected 12, but got {features.shape[0]}.")
    else:
        prediction = model.predict([features])
        result = "Approved" if prediction[0] == 1 else "Rejected"
        st.subheader(f"Loan Status: {result}")

        # Display prediction probability
        probability = model.predict_proba([features])[0][1]
        st.write(f"Prediction Probability: {probability * 100:.2f}%")

