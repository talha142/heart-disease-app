import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import io

st.set_page_config(page_title="Heart Disease App", layout="wide")

# Function to load dataset
@st.cache_data
def load_data(file=None):
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv('heart.csv')  # Default dataset (you can change this)
    return df

# App navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🤖 Prediction", "📈 Classification", "ℹ️ About"])

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
df = load_data(uploaded_file) if uploaded_file else load_data()

# ---------------------- HOME ----------------------
if page == "🏠 Home":
    st.title("❤️ Heart Disease Risk Prediction App")
    st.markdown("""
    Welcome to the **Heart Disease Prediction App**. This tool helps users identify potential heart disease risks
    using machine learning algorithms and patient health data.

    ### 👩‍⚕️ Common Symptoms
    - Chest pain or discomfort
    - Shortness of breath
    - Fatigue or dizziness

    ### 🎯 Affected Age Groups
    - Most prevalent in people aged **45 and above**
    - Risk increases with lifestyle factors like **smoking**, **poor diet**, and **lack of physical activity**

    ### ✅ Prevention Tips
    - Maintain a healthy weight
    - Regular physical exercise
    - Avoid tobacco and manage cholesterol/BP levels

    ### 📊 Age Group & Prevalence (Static Table)
    """)

    age_data = {
        "Age Group": ["30-39", "40-49", "50-59", "60+"],
        "Prevalence (%)": [5, 12, 21, 35]
    }
    st.table(pd.DataFrame(age_data))

# ---------------------- EDA ----------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if df is not None:
        st.subheader("Preview of Dataset")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("Missing Value Analysis")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)

        st.subheader("Feature Distributions by Disease")

        if 'age' in df.columns and 'TenYearCHD' in df.columns:
            st.markdown("**Age vs Disease (Boxplot)**")
            plt.figure()
            sns.boxplot(data=df, x='TenYearCHD', y='age')
            st.pyplot(plt)

        if 'sex' in df.columns and 'TenYearCHD' in df.columns:
            st.markdown("**Gender vs Disease (Barplot)**")
            plt.figure()
            sns.countplot(data=df, x='sex', hue='TenYearCHD')
            st.pyplot(plt)

        if 'cigsPerDay' in df.columns:
            st.markdown("**Smoking (cigs/day) vs Disease (Boxplot)**")
            plt.figure()
            sns.boxplot(data=df, x='TenYearCHD', y='cigsPerDay')
            st.pyplot(plt)

        if 'totChol' in df.columns and 'sysBP' in df.columns:
            st.markdown("**Cholesterol & Blood Pressure**")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.boxplot(data=df, x='TenYearCHD', y='totChol', ax=ax[0])
            sns.boxplot(data=df, x='TenYearCHD', y='sysBP', ax=ax[1])
            st.pyplot(fig)

    else:
        st.warning("No dataset found. Please upload a CSV.")

# ---------------------- PREDICTION ----------------------
elif page == "🤖 Prediction":
    st.title("🤖 Heart Disease Risk Prediction")

    if df is not None and 'TenYearCHD' in df.columns:
        # User Inputs
        st.subheader("Input Patient Details")

        gender = st.selectbox("Gender", [0, 1])
        age = st.slider("Age", 30, 80, 50)
        cigs = st.slider("Cigarettes per Day", 0, 40, 10)
        BPMeds = st.selectbox("BPMeds", [0, 1])
        stroke = st.selectbox("Stroke History", [0, 1])
        hypertension = st.selectbox("Hypertension", [0, 1])
        totChol = st.slider("Total Cholesterol", 100, 400, 200)
        sysBP = st.slider("Systolic BP", 90, 200, 120)
        BMI = st.slider("BMI", 15.0, 40.0, 25.0)

        model_choice = st.selectbox("Select ML Model", ["Random Forest", "Decision Tree", "Gradient Boosting"])

        input_df = pd.DataFrame([[gender, age, cigs, BPMeds, stroke, hypertension, totChol, sysBP, BMI]],
                                columns=['sex', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
                                         'totChol', 'sysBP', 'BMI'])

        # Prepare data
        X = df[['sex', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
                'totChol', 'sysBP', 'BMI']]
        y = df['TenYearCHD']
        X = X.dropna()
        y = y.loc[X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        if model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = GradientBoostingClassifier()

        model.fit(X_train, y_train)
        prediction = model.predict(input_df)[0]
        accuracy = accuracy_score(y_test, model.predict(X_test))

        if st.button("Predict"):
            if prediction == 1:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")

            st.info(f"Model Accuracy: {accuracy:.2%}")
    else:
        st.warning("Required columns not found in the dataset!")

# ---------------------- CLASSIFICATION ----------------------
elif page == "📈 Classification":
    st.title("📈 Model Classification Report")

    if df is not None and 'TenYearCHD' in df.columns:
        X = df[['sex', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
                'totChol', 'sysBP', 'BMI']].dropna()
        y = df['TenYearCHD'].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Classification Report")
        st.code(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        st.pyplot(plt)
    else:
        st.warning("Dataset or target column missing!")

# ---------------------- ABOUT ----------------------
elif page == "ℹ️ About":
    st.title("ℹ️ Project Overview")

    st.markdown("""
    ### 🧠 Goal
    This project predicts the **10-year risk of heart disease** using clinical and lifestyle features.

    ### 🛠 Tools & Technologies
    - **Python**
    - **Streamlit** for app interface
    - **scikit-learn** for machine learning
    - **Pandas, Seaborn, Matplotlib** for EDA

    ### 📁 Dataset Source
    - Framingham Heart Study Dataset
    - Or any uploaded custom dataset with required features

    
    """)

