import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Heart Disease ML App", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Classification", "About"])
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna()
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if "TenYearCHD" not in df.columns:
        st.error("Dataset must include 'TenYearCHD' column as the target.")
        st.stop()

    X = df.drop("TenYearCHD", axis=1)
    y = df["TenYearCHD"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

else:
    st.warning("👈 Please upload your dataset to continue.")
    st.stop()

# ------------------ Page 1: Home ------------------
if page == "Home":
    st.title("💖 Heart Disease Prediction App")

    st.markdown("""
    ### 🩺 What is Heart Disease?
    Heart disease involves narrowed or blocked blood vessels that can lead to a heart attack, chest pain, or stroke.

    ### ⚠️ Common Symptoms
    - Chest pain
    - Shortness of breath
    - Pain in neck, jaw, or back
    - Cold sweat, fatigue

    ### 🧓 Age Group Heart Disease Rate
    """)
    
    df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80], 
                             labels=['20–30', '31–40', '41–50', '51–60', '61–70', '71–80'])
    group_stats = df.groupby('age_group')["TenYearCHD"].mean().reset_index()
    group_stats.columns = ['Age Group', 'Heart Disease Rate']
    st.dataframe(group_stats)

    st.markdown("""
    ### 🛡️ How to Prevent Heart Disease
    - Maintain a healthy weight
    - Eat a balanced diet
    - Avoid smoking
    - Control blood pressure and cholesterol
    - Stay physically active
    """)

# ------------------ Page 2: EDA ------------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("🔍 First 5 Rows")
    st.dataframe(df.head())

    st.subheader("📋 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("📌 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Bar Plot: Heart Disease by Gender (if available)")
    if 'male' in df.columns:
        fig, ax = plt.subplots()
        sns.barplot(x='male', y='TenYearCHD', data=df, ax=ax)
        st.pyplot(fig)

    st.subheader("📈 Line Chart: Age vs Heart Disease Rate")
    line_data = df.groupby("age")["TenYearCHD"].mean()
    st.line_chart(line_data)

# ------------------ Page 3: Prediction ------------------
elif page == "Prediction":
    st.title("🔮 Predict Heart Disease Risk")

    st.sidebar.subheader("Select ML Model")
    model_name = st.sidebar.selectbox("Choose Model", 
                                      ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.markdown("### 🧾 Enter Patient Details")
    user_input = {}

    for col in X.columns:
        if X[col].nunique() < 10 and X[col].dtype in [np.int64, np.int32, object]:
            options = X[col].unique().tolist()
            user_input[col] = st.selectbox(f"{col}", options)
        else:
            left, right = st.columns(2)
            with left:
                user_input[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("🚨 Predict"):
        pred = model.predict(input_df)[0]
        st.subheader("📢 Prediction Result:")
        if pred == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
        st.info(f"Model Accuracy: {acc * 100:.2f}%")

# ------------------ Page 4: Classification ------------------
elif page == "Classification":
    st.title("📋 Classification Evaluation")

    eval_set = st.sidebar.radio("Evaluate on:", ["Training Set", "Testing Set"])
    model_name = st.sidebar.selectbox("Choose Model", 
                                      ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()

    model.fit(X_train, y_train)

    if eval_set == "Training Set":
        y_eval = y_train
        y_pred = model.predict(X_train)
    else:
        y_eval = y_test
        y_pred = model.predict(X_test)

    st.subheader("🧾 Classification Report")
    report = classification_report(y_eval, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap="Blues"))

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Reds", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    st.pyplot(fig)

# ------------------ Page 5: About ------------------
elif page == "About":
    st.title("ℹ️ About")

    st.markdown("""
    ### 💡 Project Overview
    A user-friendly heart disease prediction tool powered by machine learning and Streamlit.

    ### 🧰 Technologies Used
    - Python
    - Streamlit
    - Pandas, NumPy
    - Matplotlib, Seaborn
    - scikit-learn

    ### 📂 Dataset Source
    This application uses datasets from [Kaggle – Framingham Heart Study](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset)

    ⚠️ **Note**: This app is for educational and demonstration purposes only.
    """)
