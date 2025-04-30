import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd 
import pickle

## Load trained Model

model = tf.keras.models.load_model('model.h5')


with open('label_encoder_gender.pkl', 'rb') as file:
    Label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## Set a page configuration

st.set_page_config(
    page_title= 'Customer Churn Prediction',
    page_icon= "logo.png",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .fancy-container{
        border: 2px solid #4A90E2;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }       
    </style>
""", unsafe_allow_html=True)

## Streamlit App
st.title('Customer Churn Prediction')


st.markdown('<div class="fancy-container">', unsafe_allow_html=True)
## User input
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', Label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92)
    balance = st.number_input('Balance')
    credit_score = st.number_input('Credit Score')
with col2:
    estimated_salary = st.number_input('Estimated Salary')
    tenure = st.slider('Tenure', 0, 10)
    number_of_products = st.slider('Number Of Products', 1, 4)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active_member = st.selectbox('Is an Active Member', [0,1])
st.markdown('</div>', unsafe_allow_html=True)

## Prepare the Input Data 

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [Label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [number_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

## one_hot Encode Geography

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


## Combine one_hot encoding columns with data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale the input Data

input_data_scaled = scaler.transform(input_data)

## Predict Churn

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"Churn Probability : {prediction_proba:.2f}")

    with col2:

        if prediction_proba > 0.5:
            st.write('The Customer is likely to Churn')
        else:
            st.write('The Customer is not likely to Churn')



