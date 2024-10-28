import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time

# Load and preprocess the dataset
loan_data = pd.read_csv(r"C:\Users\aaras\OneDrive\Desktop\CS Work\Loan-Prediction-Tool\loan_approval_dataset.csv")
loan_data.columns = loan_data.columns.str.strip()
for col in loan_data.select_dtypes(include=['object']).columns:
    loan_data[col] = loan_data[col].str.strip()

# Title for Streamlit app
st.title("Loan Approval Prediction")

# User inputs for loan prediction
name = st.text_input("Enter your name:")
st.button("Submit")

input_no_of_dependents = st.slider("Select the number of people relying on your income", 0, 5)
input_education = st.selectbox("Select your education status:", ("Graduate", "Not Graduate"))
input_self_employed = st.selectbox("Are you self-employed?", ("Yes", "No"))
input_income_annum = st.slider("Select Your Annual Income", int(loan_data["income_annum"].min()), int(loan_data["income_annum"].max()))
input_loan_amount = st.slider("Select the amount you want as loan", int(loan_data["loan_amount"].min()), int(loan_data["loan_amount"].max()))
input_loan_term = st.slider("Select the term of loan", int(loan_data["loan_term"].min()), int(loan_data["loan_term"].max()))
input_cibil_score = st.slider("Select Your Cibil Score", 300, 900)
input_residential_assets_value = st.slider("Select your residential assets value", 10000, int(loan_data["commercial_assets_value"].max()))
input_commercial_assets_value = st.slider("Select your commercial assets value", int(loan_data["commercial_assets_value"].min()), int(loan_data["commercial_assets_value"].max()))
input_luxury_assets_value = st.slider("Select your luxury assets value", int(loan_data["luxury_assets_value"].min()), int(loan_data["luxury_assets_value"].max()))
input_bank_asset_value = st.slider("Select your bank asset value", int(loan_data["bank_asset_value"].min()), int(loan_data["bank_asset_value"].max()))

# Splitting data for training
X = loan_data.drop(columns=['loan_status', 'loan_id'])
y = loan_data['loan_status']

# Encode categorical columns
education_encoder = LabelEncoder()
self_employed_encoder=LabelEncoder()
X['education'] = education_encoder.fit_transform(X['education'])
X['self_employed'] = self_employed_encoder.fit_transform(X['self_employed'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Create user input data as DataFrame
user_input_df = pd.DataFrame({
    "no_of_dependents": [input_no_of_dependents],
    "education": [input_education],
    "self_employed": [input_self_employed],
    "income_annum": [input_income_annum],
    "loan_amount": [input_loan_amount],
    "loan_term": [input_loan_term],
    "cibil_score": [input_cibil_score],
    "residential_assets_value": [input_residential_assets_value],
    "commercial_assets_value": [input_commercial_assets_value],
    "luxury_assets_value": [input_luxury_assets_value],
    "bank_asset_value": [input_bank_asset_value]
}, columns=X.columns)

# Encode and scale user input data
user_input_df['education'] = education_encoder.transform(user_input_df['education'])
user_input_df['self_employed'] = self_employed_encoder.transform(user_input_df['self_employed'])
user_input_scaled = scaler.transform(user_input_df)

# Predict loan approval
prediction = knn.predict(user_input_scaled)

if st.button("Predict Loan Status"):
    with st.spinner("Analyzing the loan data..."):
        time.sleep(2)
        if  prediction[0] == 'Approved':
            st.write('Congratulations',name,'! Your Loan has been approved by the bank.')
        else:
            st.write('Sorry',name,'! Your Loan has been rejected by the bank.')




