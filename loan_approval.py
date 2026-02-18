import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")
st.title("Loan Approval Prediction & Analysis")

# ======================
# Load Data
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ======================
# Handle Missing Values
# ======================
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# ======================
# EDA
# ======================
st.subheader("Class Distribution")
classes_count = df["Loan_Approved"].value_counts()

fig1 = plt.figure()
plt.pie(classes_count, labels=["No", "Yes"], autopct="%1.1f%%")
plt.title("Is Loan Approved?")
st.pyplot(fig1)

st.subheader("Income Distribution")
fig2 = plt.figure()
sns.histplot(data=df, x="Applicant_Income", bins=20)
st.pyplot(fig2)

# ======================
# Drop ID Column
# ======================
df = df.drop("Applicant_ID", axis=1)

# ======================
# Encoding
# ======================
le = LabelEncoder()
df["Education_Level"] = le.fit_transform(df["Education_Level"])
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

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

# ======================
# Correlation Heatmap
# ======================
st.subheader("Correlation Heatmap")

num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols.corr()

fig3 = plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
st.pyplot(fig3)

# ======================
# Train Test Split
# ======================
x = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ======================
# Model Training
# ======================
@st.cache_resource
def train_models(x_train_scaled, y_train):
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(x_train_scaled, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train_scaled, y_train)

    nb_model = GaussianNB()
    nb_model.fit(x_train_scaled, y_train)

    return log_model, knn_model, nb_model

log_model, knn_model, nb_model = train_models(x_train_scaled, y_train)

st.subheader("Model Evaluation Results")

# Logistic Regression
y_pred_log = log_model.predict(x_test_scaled)
st.write("### Logistic Regression")
st.write("Precision:", precision_score(y_test, y_pred_log))
st.write("Recall:", recall_score(y_test, y_pred_log))
st.write("F1 Score:", f1_score(y_test, y_pred_log))
st.write("Accuracy:", accuracy_score(y_test, y_pred_log))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_log))

# KNN
y_pred_knn = knn_model.predict(x_test_scaled)
st.write("### KNN")
st.write("Precision:", precision_score(y_test, y_pred_knn))
st.write("Recall:", recall_score(y_test, y_pred_knn))
st.write("F1 Score:", f1_score(y_test, y_pred_knn))
st.write("Accuracy:", accuracy_score(y_test, y_pred_knn))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_knn))

# Naive Bayes
y_pred_nb = nb_model.predict(x_test_scaled)
st.write("### Naive Bayes")
st.write("Precision:", precision_score(y_test, y_pred_nb))
st.write("Recall:", recall_score(y_test, y_pred_nb))
st.write("F1 Score:", f1_score(y_test, y_pred_nb))
st.write("Accuracy:", accuracy_score(y_test, y_pred_nb))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred_nb))
