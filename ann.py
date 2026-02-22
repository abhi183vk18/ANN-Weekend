import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
import pickle
import streamlit as st


####load the model file


model=load_model("model.h5")


#### read the pickle files

with open ("label_gender_encode.pkl","rb") as file:
    label_gender_encode=pickle.load(file)

with open ("one_hot_geo.pkl","rb") as file:
    one_hot_geo=pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)



### streamlit app


st.title("Customer churn prediction")

geography=st.selectbox('Geography',one_hot_geo.categories_[0])
gender=st.selectbox("Gender",label_gender_encode.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])



# Example input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_gender_encode.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded=one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


## scaling the input data

input_scaled=scaler.transform(input_data)

#### prdicting the output


predict=model.predict(input_scaled)

predict_prob=predict[0][0]


if predict_prob >0.5:
    st.write("The customer will leave")
else:
    st.write("The customer will stay")