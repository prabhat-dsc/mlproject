
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

url = "mongodb+srv://prabhatdsc487:prabha123@app-cluster.9joou.mongodb.net/?retryWrites=true&w=majority&appName=app-cluster"
# Create a new client and connect to the server
client = MongoClient(url, server_api=ServerApi('1'))

db=client['Aimind']
collection=db['student_score']

def load_data():
    with open('student\student_final_model_1.pkl','rb') as file:
        model,scaler,le=pickle.load(file)
    return model,scaler,le

def data_preprocess(data):
    model,scaler,le=load_data()
    data['Extracurricular Activities']=le.transform([data['Extracurricular Activities']])
    df=pd.DataFrame(data)
    scaled_data=scaler.transform(df)
    return scaled_data

def model_prediction(data):
    model,scaler,le=load_data()
    process_data=data_preprocess(data)
    prediction=model.predict(process_data)
    return prediction

def main():
    st.title('Machine Learning Prediction App')
    st.header('Enter the input Value: ')
    user_name = st.text_input("Enter Your Name")
    Hours_Studied=st.number_input('Hours Studied',min_value=1 ,max_value=9,value=1)
    Previous_Scores=st.number_input('Previous Scores',min_value=40 ,max_value=99,value=45)
    Extracurricular_Activities=st.selectbox('Extracurricular Activities',['Yes','No'])
    Sleep_Hours = st.number_input('Sleep Hours',min_value=4 ,max_value=9,value=4)
    Sample_Ques_Papers_Practiced=st.number_input('Sample Question Papers Practiced',min_value=0 ,max_value=9,value=0)
    
    if st.button('Predict'):
        if not user_name:
            st.warning("⚠️ Please enter your name before making a prediction.")
        else:    
            user_data={
                'Hours Studied':Hours_Studied,
                'Previous Scores':Previous_Scores,
                'Extracurricular Activities':Extracurricular_Activities,
                'Sleep Hours':Sleep_Hours,
                'Sample Question Papers Practiced':Sample_Ques_Papers_Practiced
            }
            pred=round(model_prediction(user_data)[0],2)
            st.success(f'Prediction:: {pred}')
            user_data['prediction']=float(pred)
            user_data['Extracurricular Activities']=int(user_data['Extracurricular Activities'])
            user_data['user_name']=user_name
            collection.insert_one(user_data)
            st.success(f'Thank You !!')

        

if __name__=='__main__':
    main()




