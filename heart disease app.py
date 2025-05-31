import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Risk App", layout="wide")

# Load default dataset
def load_data():
    df = pd.read_csv("framingham_1.csv")
    return df

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Home", "EDA", "Prediction", "Classification", "About"])

# Home Page
if page == "Home":
    st.title("Heart Disease Risk Prediction Dashboard")
    st.write("""
        Welcome to the Heart Disease Risk Prediction Dashboard. 
        Navigate through the sidebar to explore the dataset, perform exploratory data analysis (EDA), 
        predict disease based on input, and evaluate machine learning models.
    """)

# EDA Page
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_data()

    st.subheader("Cleaned Dataset")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0])

    if missing.sum() > 0:
        if st.button("Drop Missing Rows"):
            df = df.dropna()
            st.success("Missing values removed!")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Age vs Target")
    fig, ax = plt.subplots()
    sns.boxplot(x='TenYearCHD', y='age', data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Gender vs Risk")
    gender_risk = df.groupby("male")["TenYearCHD"].mean()
    st.bar_chart(gender_risk)

    st.subheader("Smoking & Alcohol Impact")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x="currentSmoker", y="TenYearCHD", data=df, ax=ax[0])
    sns.barplot(x="BPMeds", y="TenYearCHD", data=df, ax=ax[1])
    ax[0].set_title("Smoking Impact")
    ax[1].set_title("BP Meds Impact")
    st.pyplot(fig)

    st.subheader("BP & Cholesterol vs Disease")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="TenYearCHD", y="sysBP", data=df, ax=ax[0])
    sns.boxplot(x="TenYearCHD", y="totChol", data=df, ax=ax[1])
    ax[0].set_title("Systolic BP vs Disease")
    ax[1].set_title("Cholesterol vs Disease")
    st.pyplot(fig)

# Prediction Page
elif page == "Prediction":
    st.title("Prediction")
    try:
        df = load_data()
        df = df.dropna()
        X = df.drop("TenYearCHD", axis=1)
        y = df["TenYearCHD"]

        user_input = {}
        st.sidebar.header("Enter Patient Data")
        for col in X.columns:
            user_input[col] = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()))

        model = RandomForestClassifier()
        model.fit(X, y)

        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.write("## Prediction:", "High Risk" if prediction == 1 else "Low Risk")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Classification Page
elif page == "Classification":
    st.title("Classification Evaluation")
    df = load_data()
    df = df.dropna()
    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]

    st.sidebar.subheader("Choose Classifier")
    classifier_name = st.sidebar.selectbox("Model", ("Random Forest", "Logistic Regression"))
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    if classifier_name == "Random Forest":
        clf = RandomForestClassifier()
    else:
        clf = LogisticRegression(max_iter=1000)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

# About Page
elif page == "About":
    st.title("About")
    st.write("""
        This app is created for educational purposes. 
        It helps in understanding EDA and ML model performance on heart disease prediction using the Framingham dataset.
    """)
