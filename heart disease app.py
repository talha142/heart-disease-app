# 🩺 Heart Disease Prediction App with Streamlit

## 📌 Sidebar Navigation Structure
Each page will be structured individually, and navigation will be handled using the sidebar.

---

### ✅ `main.py` (Main Navigation Entry Point)
```python
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Classification", "About"])

if page == "Home":
    import Home
elif page == "EDA":
    import EDA
elif page == "Prediction":
    import Prediction
elif page == "Classification":
    import Classification
elif page == "About":
    import About
```

---

### 🏠 `Home.py`
```python
import streamlit as st

st.title("Understanding Heart Disease")

st.header("What is Heart Disease?")
st.write("""
Heart disease refers to various types of conditions that can affect heart function. These types include:
coronary artery disease, arrhythmias, and congenital heart defects. It occurs when the heart's blood
supply is blocked or interrupted by a build-up of fatty substances.

Understanding the symptoms and risk factors of heart disease is essential for early diagnosis and prevention.
""")

st.subheader("Common Symptoms")
st.markdown("""
- Chest pain or discomfort
- Shortness of breath
- Fatigue
- Irregular heartbeat
- Swelling in legs or ankles
""")

st.subheader("Risk Factors / Conditions Leading to Heart Disease")
st.markdown("""
- High blood pressure
- High cholesterol
- Diabetes
- Smoking
- Obesity
- Sedentary lifestyle
- Stress
""")

st.subheader("Global Age Group Impact")
st.markdown("Visualize with a bar chart or table (placeholder here).")

st.subheader("How to Prevent Heart Disease")
st.markdown("""
- Regular exercise
- Healthy diet
- Avoid smoking
- Stress management
""")
```

---

### 📊 `EDA.py`
```python
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os

st.title("📊 Exploratory Data Analysis")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

@st.cache_data
def load_data():
    default_path = "framingham_1.csv"
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    else:
        st.warning("Default file not found.")
        return None

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

if df is not None:
    st.subheader("📋 Raw Data")
    st.dataframe(df.head())

    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    st.subheader("🧼 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("🔎 Visualizations")

    if 'age' in df.columns and 'TenYearCHD' in df.columns:
        st.markdown("**Age vs Heart Disease**")
        fig1 = px.histogram(df, x="age", color="TenYearCHD", barmode="group")
        st.plotly_chart(fig1)

    if 'male' in df.columns and 'TenYearCHD' in df.columns:
        st.markdown("**Gender vs Risk**")
        fig2 = px.histogram(df, x="male", color="TenYearCHD", barmode="group")
        st.plotly_chart(fig2)

    if 'cigsPerDay' in df.columns:
        st.markdown("**Cigarettes Per Day Distribution**")
        fig3 = px.histogram(df, x="cigsPerDay")
        st.plotly_chart(fig3)

    if 'totChol' in df.columns and 'TenYearCHD' in df.columns:
        st.markdown("**Cholesterol vs Heart Disease**")
        fig4 = px.box(df, x="TenYearCHD", y="totChol", points="all")
        st.plotly_chart(fig4)
else:
    st.error("Please upload a dataset or ensure 'framingham_1.csv' exists.")
```

---

### 🤖 `Prediction.py`
```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🤖 Heart Disease Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("framingham_1.csv")

df = load_data()

# Select top 10 features
features = ['male', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
            'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI']
X = df[features].dropna()
y = df.loc[X.index, 'TenYearCHD']

# Sidebar inputs
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

st.sidebar.header("Input Features")
user_input = {feature: st.sidebar.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean())) for feature in features}
input_df = pd.DataFrame([user_input])

# Model initialization
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_name == "Random Forest":
    model = RandomForestClassifier()
else:
    model = GradientBoostingClassifier()

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Prediction and results
prediction = model.predict(input_df)[0]
result = "At Risk" if prediction == 1 else "Not at Risk"

st.subheader("🔍 Prediction Result")
st.write(f"**Result:** {result}")

st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```
