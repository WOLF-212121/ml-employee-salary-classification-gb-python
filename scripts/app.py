import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# loading saved model
with open('best_model.pkl','rb') as file:
 loaded_model = pickle.load(file)

st.set_page_config(page_title="Employee Salary Classification", page_icon = "🤑", layout = 'centered')

st.title("🤑 EMPLOYEES SALARY PREDICTION APP")
st.markdown("Develop a predictive model to classify whether an employee’s salary exceeds $50,000 or not based on their input features.")

# sidebar inputs
st.sidebar.header("ENTER EMPLOYEE DETAILS")


age = st.sidebar.slider("AGE", 18,78,30)
workclass = st.sidebar.selectbox("WORKCLASS",['Federal-gov', 'Local-gov', 'Never-worked',
            'Not-Specified', 'Private','Self-emp-inc', 'Self-emp-not-inc', 'State-gov',
            'Without-pay'])
education = st.sidebar.selectbox("EDUCATION LEVEL", ['10th', '11th', '12th', '9th',
            'Assoc-acdm', 'Assoc-voc', 'Bachelors','Doctorate', 'HS-grad', 'Masters',
            'Prof-school', 'Some-college'])
occupation = st.sidebar.selectbox("JOB ROLE", ['Adm-clerical','Armed-Forces',
            'Craft-repair', 'Exec-managerial','Farming-fishing', 'Handlers-cleaners',
            'Machine-op-inspct', 'Not-Defined', 'Other-service', 'Priv-house-serv',
            'Prof-specialty', 'Protective-serv','Sales', 'Tech-support', 'Transport-moving'
            ])
hours_per_week = st.sidebar.slider("HOURS-PER-WEEK", 1,80,45)
native_country = st.sidebar.selectbox("NATIVE COUNTRY", ['Cambodia', 'Canada', 'China',
                 'Columbia', 'Cuba', 'Dominican-Republic',
                 'Ecuador', 'El-Salvador', 'England', 'France','Germany','Greece','Guatemala'
                 'Haiti','Holand-Netherlands','Honduras','Hong','Hungary','India','Iran',
                 'Ireland','Italy','Jamaica','Japan','Laos','Mexico','Nicaragua','Others',
                 'Outlying-US(Guam-USVI-etc)','Peru','Philippines','Poland','Portugal',
                 'Puerto-Rico','Scotland','South','Taiwan','Thailand','Trinadad&Tobago',
                 'United-States','Vietnam','Yugoslavia'])


# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'workclass' : [workclass],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'native-country' : [native_country]
})

st.write("### 🔎 INPUT DATA")
st.write(input_df)

# Predict button
if st.button("PREDICT SALARY CLASS"):
  for i in input_df.columns:
    if input_df[i].dtypes==object:
      input_df[i] = le.fit_transform(input_df[i])
  prediction = loaded_model.predict(input_df)
  st.success(f"✅ PREDICTION : {prediction}")

# Batch prediction
st.markdown("---")
st.markdown("## 📁 BATCH PREDICTION")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
  batch_data = pd.read_csv(uploaded_file)
  st.write("Upload data preview :", batch_data.head())
  batch_pred = loaded_model.predict(batch_data)
  batch_data['Predicted Classes'] = batch_pred
  st.write("✅ Predictions:")
  st.write(batch_data.head())
  csv = batch_data.to_csv(index=False).encode('utf-8')
  st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
