# Overview
This repository contains a Streamlit application that predicts loan approval based on user-inputted financial and personal information. The app leverages <b>machine learning</b>, specifically a K-Nearest Neighbors (KNN) classifier, to determine whether a loan application is likely to be approved or rejected by the bank. The model has been trained by a existing dataset containg financial data and loan status. <strong><b> The model boasts an accuracy of 87.9%</b></strong> which has been attained by selecting proper model, scaling and manipulation of data.
# Features
<ul>
  <li>Interactive input fields for financial and personal details
  <li>Real-time prediction of loan approval status
  <li>User-friendly Streamlit interface
  <li>Data preprocessing and scaling for accurate predictions
</ul>

# Pre-Requisites

<ul>
  <li>Python 3.11 or higher</li>
  <li>Loan Dataset from any open sources
  <li>Following python packages</li>
  <ul>
    <li>streamlit</li>
    <li>scikit-learn</li>
    <li>numpy</li>
    <li>pandas</li>
</ul>

# Information about the Dataset

The loan_approval_dataset.csv used in the project contains the following data of 4269 loan applicants:
<ul>
  <li>loan_id</li>
  <p>It contains the serial number of the loan applicants.</p>
  <li>no_of_dependents</li>
  <p>It contains the number of people depending upon the income of the applicant.</p>
  <li>education</li>
  <p>It contains the education status of applicant:Graduate or Not Graduate</p>
  <li>self_employed</li>
  <p>Contains the employement status of applicant: Yes or No</p>
  <li>income_annum</li>
  <p>Contains the per annum income of the applicant</p>
  <li>loan_amount</li>
  <p>Contains the amount for which applicant has applied for</p>
  <li>loan_term</li>
  <p>Contains the time period in years  for which apllicant promises to repay the loan </p>
  <li>cibil_score</li>
  <p>Contains an integer between 300 to 900, indicates the creditworthiness of applicant based on credit history </p>
  <li>assets</li>
  <p>residential assets, commercial assets, luxury assets and bank asstes of the applicant</p>
  <li>loan_status</li>
  <p>Cotains wether the loan has been approved or rejected by the bank</p>
</ul>

# Usage

1. Run the Streamlit app:

2. Fill in the required inputs, such as your income, loan amount, number of dependents, and other financial details in the sidebar and main interface.

3. Click "Predict" to view the loan approval prediction.

# Contact Details

Name-<b>Ayush Rastogi</b>

For any queries or feedback contact me at aarastogi.220307@gmail.com


