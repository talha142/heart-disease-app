# main.py (or run with `streamlit run Home.py` from the app root)

# 1️⃣ Home.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Understanding Heart Disease", layout="wide")
st.title("Understanding Heart Disease")

st.header("What is Heart Disease?")
st.write("""
Heart disease refers to various types of conditions that can affect heart function. These include diseases of the blood vessels, heart rhythm problems (arrhythmias), and heart defects present at birth. It is one of the leading causes of death globally.

Typically, heart disease occurs when there is a buildup of plaque in the arteries, a condition known as atherosclerosis. This buildup narrows the arteries and makes it harder for blood to flow through them.
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
age_data = pd.DataFrame({
    'Age Group': ['<30', '30-45', '46-60', '61-75', '76+'],
    'Cases (%)': [5, 15, 35, 30, 15]
})
fig, ax = plt.subplots()
ax.bar(age_data['Age Group'], age_data['Cases (%)'], color='salmon')
ax.set_ylabel('Cases (%)')
ax.set_title('Global Impact of Heart Disease by Age Group')
st.pyplot(fig)

st.subheader("How to Prevent Heart Disease")
st.markdown("""
- Regular physical activity
- Healthy diet
- Avoid tobacco
- Manage stress effectively
""")

# 2️⃣ EDA.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Exploratory Data Analysis")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("framingham_1.csv")

st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("### Summary Statistics")
st.dataframe(df.describe())

st.write("### Missing Values")
st.dataframe(df.isnull().sum())

if st.checkbox("Drop missing rows"):
    df.dropna(inplace=True)

st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Bar Charts
st.write("### Age vs Heart Disease")
sns.barplot(x="age", y="TenYearCHD", data=df)
st.pyplot()

st.write("### Gender vs Heart Disease")
sns.barplot(x="male", y="TenYearCHD", data=df)
st.pyplot()

st.write("### Smoking Impact")
sns.barplot(x="cigsPerDay", y="TenYearCHD", data=df)
st.pyplot()

# 3️⃣ Prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Heart Disease Prediction")

df = pd.read_csv("framingham_1.csv")
df.dropna(inplace=True)

features = ['male', 'age', 'cigsPerDay', 'BPMeds', 'prevalentStroke',
       'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI']
X = df[features]
y = df['TenYearCHD']

model_choice = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])

input_data = []
for feature in features:
    val = st.slider(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

# Model
if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
else:
    model = GradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

if st.button("Predict"):
    result = model.predict(input_array)
    st.write("### Prediction Result:")
    st.success("At Risk" if result[0] == 1 else "Not at Risk")
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# 4️⃣ Classification.py
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Model Evaluation")

model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"])
test_size = st.slider("Test size (percentage)", 0.1, 0.5, 0.2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
else:
    model = GradientBoostingClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# 5️⃣ About.py
import streamlit as st

st.title("About This App")

st.subheader("Technologies Used")
st.markdown("""
- Python (Pandas, NumPy)
- Matplotlib & Seaborn (for visualization)
- Scikit-learn (ML Models)
- Streamlit (Web App Framework)
""")

st.subheader("ML Models Used")
st.markdown("""
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
""")

st.subheader("Resource")
st.markdown("Dataset: [Framingham Heart Study - Kaggle](https://www.kaggle.com/datasets/)")
