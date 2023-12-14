# heart-disease-model-evaluation

## Predicting heart disease by testing multiple machine learning models
**Summary**
The heart disease predictor app uses a binary classification machine learning model. Binary classification is a type of supervised learning where the goal is to categorize input data into one of two classes or categories. In this case, the model predicts whether a person has heart disease or not based on various input features.

Here are the key components of the heart disease predictor app:
**Task Type: Binary Classification**
The machine learning model is trained to predict whether an individual has heart disease or not. The target variable has two classes: 1 for "Heart Disease" and 0 for "No Heart Disease."

**Features (Input Variables):**
Various health-related features such as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, thalium stress result, etc.

**Models Used:**
Logistic Regression (best accuracy score - 88.52%).

**Training and Evaluation:**
The model is trained on a labeled dataset where each instance is associated with a target label indicating the presence or absence of heart disease. The trained model is then evaluated on a separate set of data to assess its performance.

**Evaluation Metrics:**
Common binary classification metrics such as accuracy, precision, recall, F1 score, and the area under the Receiver Operating Characteristic (ROC) curve may be used to evaluate the model's performance.

**Use Case:**
The model can be used to assist healthcare professionals in diagnosing or assessing the risk of heart disease in individuals based on their health-related information.
