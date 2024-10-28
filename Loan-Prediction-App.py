import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier




loan_data=pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\CS Work\Loan Prediction\loan_approval_dataset.csv")
loan_data.columns = loan_data.columns.str.strip()
for col in loan_data.select_dtypes(include=['object']).columns:
    loan_data[col] = loan_data[col].str.strip()


st.title("Loan Approval Prediction")

name = st.text_input("Enter your name:")
st.button("Submit")

input_no_of_dependents=st.sidebar.slider("Select the number of people relying on your income",0,5)

input_education = st.selectbox("Select your education status:",("Graduated", "Not Graduated"))

    
input_self_employed = st.selectbox("Are you self-employed? ",("Yes", "No"))


input_income_annum=st.sidebar.slider("Select Your Annual Income",int(loan_data["income_annum"].min()),int(loan_data["income_annum"].max()))

input_loan_amount=st.sidebar.slider("Select the amount you want as loan",int(loan_data["loan_amount"].min()),int(loan_data["loan_amount"].max()))

input_loan_term=st.sidebar.slider("Select the term of loan",int(loan_data["loan_term"].min()),int(loan_data["loan_term"].max()))

input_cibil_score=st.sidebar.slider("Select Your Cibil Score ",300,900)

input_residential_assets_value=st.sidebar.slider("Select your residential assets value",10000,int(loan_data["commercial_assets_value"].max()))

input_commercial_assets_value=st.sidebar.slider("Select your commercial assets value",int(loan_data["commercial_assets_value"].min()),int(loan_data["commercial_assets_value"].max()))

input_luxury_assets_value=st.sidebar.slider("Select your luxury assets value",int(loan_data["luxury_assets_value"].min()),int(loan_data["luxury_assets_value"].max()))

input_bank_asset_value=st.sidebar.slider("Select your bank asset value",int(loan_data["bank_asset_value"].min()),int(loan_data["bank_asset_value"].max()))

X = loan_data.drop(columns=['loan_status', 'loan_id'])
y = loan_data['loan_status']

label_encoder = LabelEncoder()
X['education'] = label_encoder.fit_transform(X['education'])
X['self_employed'] = label_encoder.fit_transform(X['self_employed'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

user_input_df=pd.DataFrame({
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
    "bank_asset_value": [input_bank_asset_value],
},columns=X.columns)


user_input_df['education'] = label_encoder.fit_transform(user_input_df['education'])
user_input_df['self_employed'] = label_encoder.fit_transform(user_input_df['self_employed'])
user_input_scaled=scaler.fit_transform(user_input_df)


st.spinner("Examining the Data...")
st.spinner("Checking Loan Status...")

# Debug: Print user input data after encoding and scaling
st.write("User Input Data (after encoding):", user_input_df)
st.write("User Input Data (after scaling):", user_input_scaled)

# Step 1: Predict the loan status using the KNN model
predicted_loan_status = knn.predict(user_input_scaled)

# Debug: Print predicted loan status (encoded and decoded)
st.write("Predicted Loan Status (encoded):", predicted_loan_status)

# Step 2: Decode the predicted status if you used LabelEncoder for the target variable
predicted_loan_status_decoded = label_encoder.inverse_transform(predicted_loan_status)
st.write("Predicted Loan Status (decoded):", predicted_loan_status_decoded)

# Step 3: Display the result based on the predicted status
if predicted_loan_status_decoded[0] == "Approved":
    st.write("Congratulations! The Loan has been approved by the bank.")
else:
    st.write("The Loan has been rejected by the bank.")




