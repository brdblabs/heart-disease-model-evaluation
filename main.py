import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('heart-disease.csv')

# Drop target column and assign as y
X = df.drop('target', axis=1)
y = df['target']

# Load pre-trained Logistic Regression model
lr = joblib.load('saved_model.joblib')

# Streamlit app
st.title('Heart Disease Predictor')

# Center image in sidebar
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 70%;
        }
    </style>
    """, unsafe_allow_html=True
)

# Insert an image in the sidebar
st.sidebar.image('heart.jpeg')

# Sidebar with user input
st.sidebar.header('User Input Features')

def user_input_features():
    age = st.sidebar.slider('Age', min_value=29, max_value=77, value=55)
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    cp = st.sidebar.slider('Chest Pain Type', min_value=0, max_value=3, value=1)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', min_value=94, max_value=200, value=120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', min_value=126, max_value=564, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=71, max_value=202, value=150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, value=1.0)
    slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment', min_value=0, max_value=2, value=1)
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, value=1)
    thal = st.sidebar.slider('Thalassemia', min_value=0, max_value=7, value=1)

    data = {
        'age': age,
        'sex': 1 if sex == 'male' else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Initialize prediction variable
prediction = None

# Display the user input
user_input = user_input_features()
st.subheader('User Input:')
st.write(user_input)

# Button to trigger predictions
if st.button('Predict'):
    # Make predictions using the loaded Logistic Regression model
    prediction = lr.predict(user_input)

# Display the prediction
st.subheader('Prediction:')
if prediction is not None:
    if prediction[0] == 1:
        st.write('Heart Disease')
    else:
        st.write('No Heart Disease')

# Display the list of terms
st.subheader('Feature Descriptions:')
terms_dict = {
    "age": "Age in years",
    "sex": "Sex (1 = male; 0 = female)",
    "cp": "Chest pain type:\n"
          "0: Typical angina\n"
          "1: Atypical angina\n"
          "2: Non-anginal pain\n"
          "3: Asymptomatic",
    "trestbps": "Resting blood pressure (in mm Hg on admission to the hospital)\n"
                "Anything above 130-140 is typically cause for concern",
    "chol": "Serum cholestoral in mg/dl\n"
            "Serum = LDL + HDL + 0.2 * triglycerides\n"
            "Above 200 is cause for concern",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)\n"
           "'>126' mg/dL signals diabetes",
    "restecg": "Resting electrocardiographic results:\n"
               "0: Nothing to note\n"
               "1: ST-T Wave abnormality (non-normal heart beat)\n"
               "2: Possible or definite left ventricular hypertrophy (enlarged heart's main pumping chamber)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes; 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest:\n"
               "Looks at stress of heart during exercise\n"
               "Unhealthy heart will stress more",
    "slope": "The slope of the peak exercise ST segment:\n"
             "0: Upsloping (better heart rate with exercise, uncommon)\n"
             "1: Flatsloping (minimal change, typical healthy heart)\n"
             "2: Downsloping (signs of unhealthy heart)",
    "ca": "Number of major vessels (0-3) colored by flourosopy:\n"
          "Colored vessel means the doctor can see the blood passing through\n"
          "The more blood movement, the better (no clots)",
    "thal": "Thalium stress result:\n"
            "1,3: Normal\n"
            "6: Fixed defect (used to be defect but okay now)\n"
            "7: Reversible defect (no proper blood movement when exercising)",
}

for feature, description in terms_dict.items():
    st.write(f"- **{feature}**: {description}")

# Display the dataset
if st.checkbox('Show Dataset'):
    st.subheader('Raw Data')
    st.write(df)
