import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Page setup
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

# Sidebar Navigation
pages = ["Home", "EDA", "Prediction", "Classification", "About"]
page = st.sidebar.selectbox("Select Page", pages)

# Global dataset
@st.cache_data

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# HOME PAGE
if page == "Home":
    st.title("Heart Disease Prediction Web Application")
    st.markdown("""
    ### What is Heart Disease?
    Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include blood vessel diseases, such as coronary artery disease; heart rhythm problems (arrhythmias); and heart defects you're born with (congenital heart defects), among others.

    ### Common Symptoms:
    - Chest pain
    - Shortness of breath
    - Pain, numbness, weakness or coldness in your legs or arms
    - Pain in the neck, jaw, throat, upper abdomen or back

    ### Common Causes and Risk Factors:
    - High blood pressure
    - Smoking
    - Diabetes
    - Obesity
    - Lack of exercise

    ### Global Risk Table by Age Group
    """)

    data = {
        "Age Group": ["<30", "30-40", "40-50", "50-60", ">60"],
        "Prevalence %": [5, 12, 25, 35, 23]
    }
    st.table(pd.DataFrame(data))

    st.markdown("""
    ### Prevention:
    - Regular health screenings
    - Healthy eating
    - Quit smoking
    - Regular physical activity
    - Managing stress
    """)

# EDA PAGE
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.subheader("Cleaned Data Preview")
        st.write(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt.gcf())

        st.subheader("Bar Chart: Gender Distribution")
        st.bar_chart(df['male'].value_counts())

        st.subheader("Line Chart: Age vs Cholesterol")
        st.line_chart(df[['age', 'totChol']].sort_values(by='age'))

# PREDICTION PAGE
elif page == "Prediction":
    st.title("Heart Disease Prediction")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)

        # Selected features
        features = ['male', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                    'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI']

        st.subheader("Enter Input Features")
        col1, col2 = st.columns(2)
        with col1:
            male = st.selectbox("Gender", [0, 1])
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, value=5)
            BPMeds = st.selectbox("Blood Pressure Meds", [0, 1])
            prevalentStroke = st.selectbox("Prevalent Stroke", [0, 1])
        with col2:
            prevalentHyp = st.selectbox("Prevalent Hypertension", [0, 1])
            totChol = st.number_input("Total Cholesterol", min_value=100, value=200)
            sysBP = st.number_input("Systolic BP", min_value=90, value=120)
            diaBP = st.number_input("Diastolic BP", min_value=60, value=80)
            BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

        user_data = pd.DataFrame([[male, age, cigsPerDay, BPMeds, prevalentStroke,
                                   prevalentHyp, totChol, sysBP, diaBP, BMI]],
                                 columns=features)

        st.subheader("Select Model")
        model_option = st.selectbox("Choose Classifier", ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"])

        if st.button("Predict"):
            df.dropna(inplace=True)
            X = df[features]
            y = df['TenYearCHD']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_option == "Logistic Regression":
                model = LogisticRegression()
            elif model_option == "Random Forest":
                model = RandomForestClassifier()
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = GradientBoostingClassifier()

            model.fit(X_train, y_train)
            prediction = model.predict(user_data)
            accuracy = model.score(X_test, y_test)

            st.success(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
            st.info(f"Model Accuracy: {accuracy:.2f}")

# CLASSIFICATION PAGE
elif page == "Classification":
    st.title("Model Classification Report")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        df.dropna(inplace=True)
        target = 'TenYearCHD'

        st.subheader("Select Model")
        model_option = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"])
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categoricals
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = GradientBoostingClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(plt.gcf())

# ABOUT PAGE
elif page == "About":
    st.title("About the Project")
    st.markdown("""
    ### Project Objective:
    This web application helps users analyze and predict the risk of heart disease using machine learning models. It offers exploratory data analysis, prediction with different models, and performance evaluation tools.

    ### Technologies Used:
    - Python
    - Streamlit
    - Pandas, NumPy
    - Matplotlib, Seaborn
    - Scikit-learn

    ### Data Source:
    - [Kaggle - Framingham Heart Study Dataset](https://www.kaggle.com/datasets/)
    """)
