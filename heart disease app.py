import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Heart Disease Web App", layout="wide")

# Sidebar Navigation
pages = ["Home", "EDA", "Prediction", "Classification", "About"]
selected = st.sidebar.selectbox("Select a Page", pages)

# File uploader
df = None
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# --- HOME PAGE ---
if selected == "Home":
    st.title("Web Application for Heart Disease using Machine Learning")

    st.subheader("💓 What is Heart Disease?")
    st.write("""
    Heart disease refers to various types of heart conditions, including coronary artery disease,
    arrhythmias, and congenital heart defects. It is a leading cause of death globally.
    """)

    st.subheader("🩺 Symptoms of Heart Disease")
    st.markdown("""
    - Chest pain
    - Shortness of breath
    - Fatigue
    - Pain in the neck, jaw, throat, or back
    - Irregular heartbeat
    """)

    st.subheader("📊 Age Groups Most Affected by Heart Disease")
    st.markdown("Heart disease prevalence increases with age. Below is a sample table:")

    age_data = pd.DataFrame({
        "Age Group": ["0-20", "21-40", "41-60", "61-80", "80+"],
        "Prevalence (%)": [1, 5, 20, 35, 50]
    })
    st.table(age_data)

    st.subheader("💡 Prevention Tips")
    st.markdown("""
    - Avoid smoking
    - Maintain a healthy weight
    - Exercise regularly
    - Eat a balanced diet
    - Monitor blood pressure and cholesterol
    - Reduce stress
    """)

# --- EDA PAGE ---
elif selected == "EDA":
    st.title("Exploratory Data Analysis")

    if df is not None:
        st.subheader("Cleaned Dataset")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())

        st.subheader("Feature Distribution")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        selected_feature = st.selectbox("Select Feature for Bar Plot", numeric_cols)
        plt.figure()
        sns.histplot(df[selected_feature], kde=True)
        st.pyplot(plt.gcf())

        st.subheader("Line Chart")
        line_feature = st.selectbox("Select Feature for Line Chart", numeric_cols)
        st.line_chart(df[line_feature])
    else:
        st.warning("Please upload a dataset to proceed.")

# --- PREDICTION PAGE ---
elif selected == "Prediction":
    st.title("Heart Disease Prediction")
    st.markdown("Enter patient details below to predict the likelihood of heart disease:")

    col1, col2 = st.columns(2)

    with col1:
        male = st.selectbox("Gender", options=["Female", "Male"])
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0)
        BPMeds = st.selectbox("On Blood Pressure Medications", options=["No", "Yes"])
        prevalentStroke = st.selectbox("History of Stroke", options=["No", "Yes"])

    with col2:
        prevalentHyp = st.selectbox("Hypertension", options=["No", "Yes"])
        totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=0, max_value=1000, value=200)
        sysBP = st.number_input("Systolic BP (mm Hg)", min_value=0, max_value=300, value=120)
        diaBP = st.number_input("Diastolic BP (mm Hg)", min_value=0, max_value=200, value=80)
        BMI = st.number_input("Body Mass Index", min_value=0.0, max_value=100.0, value=25.0)

    input_dict = {
        'male': 1 if male == "Male" else 0,
        'age': age,
        'cigsPerDay': cigsPerDay,
        'BPMeds': 1 if BPMeds == "Yes" else 0,
        'prevalentStroke': 1 if prevalentStroke == "Yes" else 0,
        'prevalentHyp': 1 if prevalentHyp == "Yes" else 0,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI
    }

    input_df = pd.DataFrame([input_dict])

    st.subheader("Select Prediction Model")
    model_name = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"])

    if st.button("Predict"):
        if df is not None:
            X = df[['male', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                    'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI']]
            y = df['TenYearCHD']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_name == "Logistic Regression":
                model = LogisticRegression()
            elif model_name == "Random Forest":
                model = RandomForestClassifier()
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = GradientBoostingClassifier()

            model.fit(X_train, y_train)
            prediction = model.predict(input_df)[0]
            accuracy = model.score(X_test, y_test)

            st.success(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
            st.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        else:
            st.warning("Please upload a dataset to proceed.")

# --- CLASSIFICATION PAGE ---
elif selected == "Classification":
    st.title("Model Evaluation: Classification Report and Confusion Matrix")

    if df is not None:
        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting"])

        X = df[['male', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
                'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI']]
        y = df['TenYearCHD']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = GradientBoostingClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(plt.gcf())
    else:
        st.warning("Please upload a dataset to proceed.")

# --- ABOUT PAGE ---
elif selected == "About":
    st.title("About This Project")
    st.markdown("""
    - **Project Goal:** Predict the likelihood of heart disease based on medical and lifestyle factors.
    - **Technologies Used:** Python, Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn
    - **Dataset Source:** [Kaggle - Framingham Heart Study Dataset](https://www.kaggle.com/datasets/)
    - **Key Features:**
        - Multi-page Streamlit app
        - User-uploaded dataset
        - EDA, ML Prediction, and Model Evaluation
    """)
