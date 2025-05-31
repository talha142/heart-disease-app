import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (replace with your actual data loading)
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")  # Your dataset path
    return df

df = load_data()

# Prepare data
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app layout
st.title("❤️ Heart Disease Prediction App")

page = st.sidebar.selectbox("Choose a page", ["Home", "Prediction"])

if page == "Home":
    st.header("💡 Heart Disease Dataset Overview")

    st.write("Here is a summary table with percentages (out of 100%) for each categorical feature:")

    # Convert categorical counts to percentage
    cat_cols = ["male", "BPMeds", "prevalentStroke", "prevalentHyp"]
    pct_df = pd.DataFrame()
    for col in cat_cols:
        counts = df[col].value_counts(normalize=True) * 100
        pct_df[col] = counts

    # Fill missing values with 0 and round
    pct_df = pct_df.fillna(0).round(2)
    pct_df.index.name = "Category"
    st.dataframe(pct_df)

    st.write("---")
    st.write("### Dataset Sample")
    st.dataframe(df.head())

elif page == "Prediction":
    st.header("🤖 Predict Heart Disease")

    st.sidebar.header("🔍 Select Model")
    model_name = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = GradientBoostingClassifier()

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.markdown("### 🧾 Enter Patient Info")

    # Numeric inputs with reasonable ranges & default mean values from df
    user_input = {}
    user_input['male'] = st.number_input("Male (0 = Female, 1 = Male)", min_value=0, max_value=1, value=int(df['male'].mean()), step=1)
    user_input['age'] = st.number_input("Age", min_value=20, max_value=80, value=int(df['age'].mean()), step=1)
    user_input['cigsPerDay'] = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=int(df['cigsPerDay'].mean()), step=1)
    user_input['BPMeds'] = st.number_input("On Blood Pressure Medication (0 = No, 1 = Yes)", min_value=0, max_value=1, value=int(df['BPMeds'].mean()), step=1)
    user_input['prevalentStroke'] = st.number_input("History of Stroke (0 = No, 1 = Yes)", min_value=0, max_value=1, value=int(df['prevalentStroke'].mean()), step=1)
    user_input['prevalentHyp'] = st.number_input("History of Hypertension (0 = No, 1 = Yes)", min_value=0, max_value=1, value=int(df['prevalentHyp'].mean()), step=1)
    user_input['totChol'] = st.number_input("Total Cholesterol", min_value=100, max_value=600, value=int(df['totChol'].mean()), step=1)
    user_input['sysBP'] = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=int(df['sysBP'].mean()), step=1)
    user_input['diaBP'] = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=150, value=int(df['diaBP'].mean()), step=1)
    user_input['BMI'] = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=float(round(df['BMI'].mean(), 1)), step=0.1)

    if st.button("🚨 Predict"):
        try:
            input_data = user_input.copy()

            # Fill missing columns if any (very rare here)
            for col in X_train.columns:
                if col not in input_data:
                    input_data[col] = X_train[col].mean()

            input_df = pd.DataFrame([input_data])
            input_df = input_df[X_train.columns]  # Ensure order

            pred = model.predict(input_df)[0]

            st.subheader("🔍 Prediction Result:")
            if pred == 1:
                st.error("⚠️ HIGH risk of heart disease.")
            else:
                st.success("✅ LOW risk of heart disease.")

            st.info(f"Model Accuracy: {acc * 100:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")
