import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🧠 Heart Disease Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("framingham_1.csv")

df = load_data()
df = df.dropna()  # Drop missing values

# Features and target
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar model selection
st.sidebar.header("🔍 Model Selection")
model_name = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

# Model selection
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_name == "Random Forest":
    model = RandomForestClassifier()
else:
    model = GradientBoostingClassifier()

# Train the selected model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# User input for prediction
st.markdown("### 🧾 Enter Patient Information")

input_data = {}
for col in X.columns:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())
    input_data[col] = st.slider(label=col, min_value=col_min, max_value=col_max, value=col_mean)

input_df = pd.DataFrame([input_data])

# Predict button
if st.button("🚨 Predict Heart Disease Risk"):
    prediction = model.predict(input_df)[0]
    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ The model predicts a HIGH risk of heart disease.")
    else:
        st.success("✅ The model predicts a LOW risk of heart disease.")
    
    st.info(f"🧮 Model Accuracy: **{accuracy * 100:.2f}%**")
