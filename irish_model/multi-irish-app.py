import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

models={
    'Decision Tree':joblib.load('dt_model.pkl'),
    'SVC':joblib.load('svc_model.pkl'),
    'Logistic Regression':joblib.load('log_reg_model.pkl')
}

def load_model():
    model,scaler=joblib.load('log_reg_model.pkl')
    return model,scaler

def data_preprocess(data,scaler):
    df=pd.DataFrame(data,index=[0])
    df_scaled=scaler.transform(df)
    return df_scaled

def model_prediction(data,model_choice):
    _,scaler=load_model()
    processed_data=data_preprocess(data,scaler)
    model=models[model_choice][0]
    prediction=model.predict(processed_data)
    return prediction

def main():
    st.title("🌸 Iris Flower Species Prediction")
    st.sidebar.header("🔍 Model Selection")

    model_choice = st.sidebar.selectbox("Select a Model for Prediction",list(models.keys()))

    user_name=st.text_input("Enter Your Name")
    sepal_length=st.slider('sepal length (cm)',min_value=4.0,max_value=8.0 ,value=4.0)
    sepal_width= st.slider('sepal width (cm)',min_value=2.0 ,max_value=5.0 ,value=2.0)
    petal_length= st.slider('petal length (cm)',min_value=1.0,max_value=7.0 ,value=1.0)
    petal_width= st.slider('petal width (cm)',min_value= 0.1,max_value=3.0 ,value=0.1)
    
    if st.button('Predict 🌸'):
        if not user_name:
            st.warning("⚠️ Please enter your name before making a prediction.")
        else:
            user_data={
                    'sepal length (cm)':sepal_length,
                    'sepal width (cm)':sepal_width,
                    'petal length (cm)':petal_length,
                    'petal width (cm)':petal_width
                }
            prediction=model_prediction(user_data,model_choice)
            label={0:'sesota',1:'versicolor',2:'virginica'}
            st.subheader("🔹 Prediction Result")
            st.success(f"Prdicted Flower Spices 🌸:: {label[prediction[0]]}")

if __name__=='__main__':
    main()