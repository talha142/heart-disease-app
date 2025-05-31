import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import f_classif

# Sidebar navigation
st.sidebar.title("Heart Disease ML Web App")
pages = ["Home", "EDA", "Prediction", "Classification", "About"]
page = st.sidebar.radio("Navigate", pages)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Data cleaning
    df = df.dropna()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Train/test split
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ANOVA F-test for top 10 features
    f_values, _ = f_classif(X_train, y_train)
    feature_scores = pd.Series(f_values, index=X.columns).sort_values(ascending=False)
    top_features = feature_scores.head(10).index.tolist()

    if page == "Home":
        st.title("🏥 Web Application for Heart Disease using ML")
        st.markdown("""
        ### 💓 What is Heart Disease?
        Heart disease describes a range of conditions that affect your heart. Diseases under the heart disease umbrella include:
        - Blood vessel diseases (e.g., coronary artery disease)
        - Heart rhythm problems (arrhythmias)
        - Heart defects you're born with (congenital heart defects)

        ### ⚠️ Symptoms
        - Chest pain
        - Shortness of breath
        - Pain, numbness, or coldness in legs or arms

        ### 🌍 Global Impact by Age Group
        """)

        age_bins = [20, 30, 40, 50, 60, 70, 80]
        df['AgeGroup'] = pd.cut(df['age'], bins=age_bins)
        st.bar_chart(df.groupby('AgeGroup')['TenYearCHD'].mean())

        st.markdown("""
        ### 🛡️ Prevention Tips
        - Eat a healthy diet
        - Exercise regularly
        - Avoid smoking
        - Control high blood pressure, cholesterol, and diabetes
        """)

    elif page == "EDA":
        st.title("📊 Exploratory Data Analysis")
        st.subheader("Raw Dataset")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("Bar Chart - Heart Disease by Gender")
        fig, ax = plt.subplots()
        df.groupby('male')['TenYearCHD'].mean().plot(kind='bar', ax=ax)
        ax.set_xlabel('Male (0=Female, 1=Male)')
        ax.set_ylabel('Heart Disease Rate')
        st.pyplot(fig)

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

        model.fit(X_train[top_features], y_train)
        y_pred = model.predict(X_test[top_features])
        acc = accuracy_score(y_test, y_pred)

        st.markdown("### 🧾 Enter Patient Details")
        user_input = {}
        for i in range(0, len(top_features), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(top_features):
                    col = top_features[i + j]
                    if df[col].nunique() < 10 and df[col].dtype in [np.int64, np.int32, object]:
                        user_input[col] = cols[j].selectbox(f"{col}", sorted(df[col].unique()))
                    else:
                        user_input[col] = cols[j].number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        input_df = pd.DataFrame([user_input])
        if st.button("🚨 Predict"):
            pred = model.predict(input_df)[0]
            st.subheader("📢 Prediction Result:")
            if pred == 1:
                st.error("⚠️ High Risk of Heart Disease")
            else:
                st.success("✅ Low Risk of Heart Disease")
            st.info(f"Model Accuracy: {acc * 100:.2f}%")

    elif page == "Classification":
        st.title("📈 Model Evaluation")
        st.subheader("Choose Model for Classification")
        clf_name = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

        if clf_name == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000)
        elif clf_name == "Decision Tree":
            clf = DecisionTreeClassifier()
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier()
        else:
            clf = GradientBoostingClassifier()

        clf.fit(X_train[top_features], y_train)
        y_pred = clf.predict(X_test[top_features])

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    elif page == "About":
        st.title("ℹ️ About Project")
        st.markdown("""
        This machine learning web app was built using:

        - Python 🐍
        - Streamlit 📺
        - Pandas, NumPy, Matplotlib, Seaborn 📊
        - Scikit-learn for ML modeling 🤖

        **Dataset Source:** [Kaggle](https://www.kaggle.com/datasets/)
        """)
else:
    st.warning("Please upload a dataset to begin.")
