import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Heart Disease Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 15px;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 20px;
    }
    h1 {
        color: #d63031;
    }
    h2 {
        color: #e17055;
    }
    .sidebar .sidebar-content {
        background-color: #2d3436;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("framingham_1.csv")
        # Clean data - example cleaning steps
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    except:
        # Return sample data if file not found
        data = {
            'age': np.random.randint(30, 70, 100),
            'sex': np.random.randint(0, 2, 100),
            'cp': np.random.randint(0, 4, 100),
            'trestbps': np.random.randint(90, 200, 100),
            'chol': np.random.randint(120, 400, 100),
            'fbs': np.random.randint(0, 2, 100),
            'restecg': np.random.randint(0, 3, 100),
            'thalach': np.random.randint(70, 200, 100),
            'exang': np.random.randint(0, 2, 100),
            'oldpeak': np.random.uniform(0, 6, 100),
            'slope': np.random.randint(0, 3, 100),
            'ca': np.random.randint(0, 4, 100),
            'thal': np.random.randint(0, 3, 100),
            'target': np.random.randint(0, 2, 100)
        }
        return pd.DataFrame(data)

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Model Evaluation", "About"])

# Home Page
if page == "Home":
    st.title("❤️ Understanding Heart Disease")
    st.image("https://images.unsplash.com/photo-1571902943202-507ec2618e8f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             width=700, caption="Heart Health Awareness")
    
    st.markdown("---")
    
    st.header("What is Heart Disease?")
    st.write("""
    Heart disease refers to various types of conditions that affect the heart's structure and function. 
    It's the leading cause of death globally, responsible for about 1 in every 4 deaths.
    
    The most common type is coronary artery disease, which can lead to heart attacks. Other types 
    include heart failure, arrhythmias, and congenital heart defects. Heart disease develops when 
    plaque builds up in the arteries that supply blood to the heart, making it harder for blood to flow properly.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Common Symptoms")
        symptoms = [
            "Chest pain or discomfort (angina)",
            "Shortness of breath",
            "Fatigue and weakness",
            "Irregular heartbeat",
            "Swelling in legs or ankles",
            "Dizziness or lightheadedness",
            "Nausea or lack of appetite"
        ]
        for symptom in symptoms:
            st.markdown(f"- {symptom}")
    
    with col2:
        st.header("Risk Factors")
        risks = [
            "High blood pressure",
            "High cholesterol",
            "Diabetes",
            "Smoking",
            "Obesity",
            "Sedentary lifestyle",
            "Stress",
            "Family history",
            "Age (men >45, women >55)",
            "Unhealthy diet"
        ]
        for risk in risks:
            st.markdown(f"- {risk}")
    
    st.markdown("---")
    
    st.header("Global Age Group Impact")
    age_data = pd.DataFrame({
        'Age Group': ['18-29', '30-39', '40-49', '50-59', '60-69', '70+'],
        'Prevalence (%)': [2, 8, 15, 25, 35, 45]
    })
    fig = px.bar(age_data, x='Age Group', y='Prevalence (%)', 
                 title="Heart Disease Prevalence by Age Group",
                 color='Age Group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.header("Prevention Tips")
    prevention = [
        ("Regular Exercise", "At least 150 minutes of moderate exercise per week"),
        ("Healthy Diet", "Focus on fruits, vegetables, whole grains, lean proteins"),
        ("Avoid Smoking", "Smoking damages blood vessels and heart"),
        ("Stress Management", "Practice relaxation techniques like meditation"),
        ("Regular Check-ups", "Monitor blood pressure, cholesterol, and diabetes"),
        ("Maintain Healthy Weight", "BMI between 18.5 and 24.9"),
        ("Limit Alcohol", "No more than 1 drink/day for women, 2 for men")
    ]
    
    for title, desc in prevention:
        with st.expander(title):
            st.write(desc)

# EDA Page
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Missing Values")
        missing = df.isnull().sum().to_frame("Missing Values")
        st.dataframe(missing)
        
        if st.button("Handle Missing Values"):
            df = df.dropna()
            st.success("Missing values dropped successfully!")
    
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
    
    st.markdown("---")
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Key Visualizations")
    
    plot_options = [
        "Age vs Heart Disease",
        "Gender vs Risk",
        "Smoking Impact",
        "Blood Pressure vs Disease",
        "Cholesterol Levels"
    ]
    selected_plot = st.selectbox("Select Visualization", plot_options)
    
    if selected_plot == "Age vs Heart Disease":
        fig = px.histogram(df, x='age', color='target', 
                          title="Age Distribution by Heart Disease Status",
                          labels={'age': 'Age', 'target': 'Heart Disease'},
                          nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected_plot == "Gender vs Risk":
        gender_map = {0: 'Female', 1: 'Male'}
        df['sex'] = df['sex'].map(gender_map)
        fig = px.pie(df, names='sex', title="Gender Distribution in Dataset")
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.bar(df, x='sex', color='target', 
                     title="Heart Disease by Gender",
                     labels={'sex': 'Gender', 'target': 'Heart Disease'})
        st.plotly_chart(fig2, use_container_width=True)
    
    elif selected_plot == "Smoking Impact":
        fig = px.box(df, x='target', y='cigsPerDay', 
                    title="Smoking Impact on Heart Disease",
                    labels={'target': 'Heart Disease', 'cigsPerDay': 'Cigarettes per Day'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected_plot == "Blood Pressure vs Disease":
        fig = px.scatter(df, x='trestbps', y='chol', color='target',
                        title="Blood Pressure vs Cholesterol by Heart Disease Status",
                        labels={'trestbps': 'Resting Blood Pressure', 
                               'chol': 'Cholesterol',
                               'target': 'Heart Disease'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected_plot == "Cholesterol Levels":
        fig = px.violin(df, y='chol', x='target', box=True,
                        title="Cholesterol Distribution by Heart Disease Status",
                        labels={'chol': 'Cholesterol', 'target': 'Heart Disease'})
        st.plotly_chart(fig, use_container_width=True)

# Prediction Page
elif page == "Prediction":
    st.title("🔮 Heart Disease Risk Prediction")
    
    # Model selection
    model_name = st.selectbox("Select Model", 
                             ["Logistic Regression", 
                              "Decision Tree", 
                              "Random Forest", 
                              "Gradient Boosting"])
    
    # Feature inputs
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
                         ["Typical Angina", 
                          "Atypical Angina", 
                          "Non-anginal Pain", 
                          "Asymptomatic"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    
    with col2:
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results", 
                              ["Normal", 
                               "ST-T Wave Abnormality", 
                               "Left Ventricular Hypertrophy"])
        thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)
    
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                           ["Upsloping", "Flat", "Downsloping"])
        ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 0)
    
    # Encode inputs
    sex = 1 if sex == "Male" else 0
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp = cp_map[cp]
    fbs = 1 if fbs == "Yes" else 0
    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    restecg = restecg_map[restecg]
    exang = 1 if exang == "Yes" else 0
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_map[slope]
    
    # Prepare input data
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]],
                            columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                    'thalach', 'exang', 'oldpeak', 'slope', 'ca'])
    
    # Train model
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier()
    
    model.fit(X_train, y_train)
    
    # Prediction button
    if st.button("Predict Heart Disease Risk"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"🚨 Prediction: At Risk (Probability: {probability*100:.2f}%)")
        else:
            st.success(f"✅ Prediction: Not at Risk (Probability: {probability*100:.2f}%)")
        
        st.subheader("Model Performance")
        accuracy = model.score(X_test, y_test)
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("📈 Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox("Select Model", 
                                ["Logistic Regression", 
                                 "Decision Tree", 
                                 "Random Forest", 
                                 "Gradient Boosting"],
                                key="eval_model")
        
        test_size = st.slider("Test Size Percentage", 10, 40, 20)
    
    with col2:
        eval_metric = st.selectbox("Evaluation Metric to Display",
                                 ["Confusion Matrix",
                                  "Classification Report",
                                  "ROC Curve"])
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    # Train model
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    st.markdown("---")
    
    if eval_metric == "Confusion Matrix":
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
    elif eval_metric == "Classification Report":
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))
        
    elif eval_metric == "ROC Curve":
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Model Performance Summary")
    
    accuracy = model.score(X_test, y_test)
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        precision = report['1']['precision']
        st.metric("Precision", f"{precision*100:.2f}%")
    
    with col2:
        recall = report['1']['recall']
        st.metric("Recall", f"{recall*100:.2f}%")
    
    with col3:
        f1 = report['1']['f1-score']
        st.metric("F1 Score", f"{f1*100:.2f}%")

# About Page
elif page == "About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    This web application is designed to help users understand heart disease, explore relevant data, 
    and predict heart disease risk using machine learning models.
    """)
    
    st.markdown("---")
    
    st.header("Technologies Used")
    
    tech_cols = st.columns(3)
    
    with tech_cols[0]:
        st.subheader("Data Processing")
        st.markdown("""
        - Python
        - Pandas
        - NumPy
        """)
    
    with tech_cols[1]:
        st.subheader("Visualization")
        st.markdown("""
        - Matplotlib
        - Seaborn
        - Plotly
        """)
    
    with tech_cols[2]:
        st.subheader("Machine Learning")
        st.markdown("""
        - Scikit-learn
        """)
    
    st.markdown("---")
    
    st.header("Machine Learning Models")
    
    model_cols = st.columns(4)
    
    with model_cols[0]:
        st.markdown("""
        **Logistic Regression**
        - Simple linear model
        - Good baseline
        """)
    
    with model_cols[1]:
        st.markdown("""
        **Decision Tree**
        - Non-linear model
        - Easy to interpret
        """)
    
    with model_cols[2]:
        st.markdown("""
        **Random Forest**
        - Ensemble method
        - Reduces overfitting
        """)
    
    with model_cols[3]:
        st.markdown("""
        **Gradient Boosting**
        - Sequential trees
        - High performance
        """)
    
    st.markdown("---")
    
    st.header("Dataset Information")
    st.markdown("""
    The dataset used in this application is the Framingham Heart Study dataset, 
    which is a long-term, ongoing cardiovascular study of residents of Framingham, Massachusetts.
    
    **Source:** [Kaggle](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset)
    """)
    
    st.markdown("---")
    
    st.header("Disclaimer")
    st.warning("""
    This application is for educational and informational purposes only. 
    It is not intended to replace professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any 
    questions you may have regarding a medical condition.
    """)
