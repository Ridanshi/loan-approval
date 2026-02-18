import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")

# =========================
# Load & Prepare Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

# Handle Missing Values
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# Drop ID
df = df.drop("Applicant_ID", axis=1)

# Encode Education + Target
le_edu = LabelEncoder()
df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])

# OneHot Encoding
cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols))

df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)

# Train-Test Split
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Model
@st.cache_resource
def train_model(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model

model = train_model(X_train_scaled, y_train)

# =========================
# USER INPUT FORM
# =========================

st.subheader("Enter Applicant Details")

age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")
dependents = st.number
