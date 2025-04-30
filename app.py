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
with st.container():
    st.markdown(
        "<h1 style='text-align: center; color: #f50707;'>Customer Churn Prediction</h1>",
        unsafe_allow_html=True
    )

with st.container():
    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] > div > div {
            border: 2px solid #ccc;
            padding: 10px;
            border-radius: 10px;
            background-color: #9fa0a1;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .custom-label {
            font-weight: bold;
            color: #2C3E50;
            font-size: 15px;
            margin-bottom: 2px;
            display: block;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<label class='custom-label'>Geography</label>", unsafe_allow_html=True)
        geography = st.selectbox('', onehot_encoder_geo.categories_[0], key='geo')

        st.markdown("<label class='custom-label'>Gender</label>", unsafe_allow_html=True)
        gender = st.selectbox('', Label_encoder_gender.classes_, key='gender')

        st.markdown("<label class='custom-label'>Age</label>", unsafe_allow_html=True)
        age = st.slider('', 18, 92, key='age')

        st.markdown("<label class='custom-label'>Balance</label>", unsafe_allow_html=True)
        balance = st.number_input('', key='balance_input')

        st.markdown("<label class='custom-label'>Credit Score</label>", unsafe_allow_html=True)
        credit_score = st.number_input('', key='credit_score_input')

    with col2:
        st.markdown("<label class='custom-label'>Estimated Salary</label>", unsafe_allow_html=True)
        estimated_salary = st.number_input('', key='salary_input')

        st.markdown("<label class='custom-label'>Tenure</label>", unsafe_allow_html=True)
        tenure = st.slider('', 0, 10, key='tenure')

        st.markdown("<label class='custom-label'>Number of Products</label>", unsafe_allow_html=True)
        number_of_products = st.slider('', 1, 4, key='products')

        st.markdown("<label class='custom-label'>Has Credit Card</label>", unsafe_allow_html=True)
        has_cr_card = st.selectbox('', [0, 1], key='credit_card')

        st.markdown("<label class='custom-label'>Is an Active Member</label>", unsafe_allow_html=True)
        is_active_member = st.selectbox('', [0, 1], key='active_member')


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
    st.markdown('<div class ="custom-box">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
       st.write(f"<h4 style='font-size: 20px; color : #051752;'>Churn Probability: {prediction_proba:.2f}</p>", unsafe_allow_html=True)
    with col2:

        if prediction_proba > 0.5:
            st.write("<h4 style='font-size: 20px; color : #7a0202;'>🛑 <strong>The Customer is likely to Churn</strong></p>", unsafe_allow_html=True)
        else:
            st.write("<h4 style='font-size: 20px; color : #2c7a02; '>✅ <strong>The Customer is not likely to Churn</strong></p>", unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)



# Inject custom CSS to target the next block

