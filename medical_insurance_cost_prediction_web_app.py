# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 00:53:26 2023

@author: HP
"""

import numpy as np 
import pickle 
import streamlit as st

loaded_model = pickle.load(open("Trained_model.sav","rb"))

def health_insurance_cost(input_data):
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    return prediction
    

def main():
    st.title("Health Insurance Cost Prediction Web App")
    
    # getting input from the user
    
    Age = st.text_input('Age')
    Sex = st.text_input('Sex (1->Male, 0->Female)')
    bmi = st.text_input("BMI")
    children = st.text_input("No of children")
    smoker = st.text_input("Smoker or not (0->non smoker, 1 -> non smoker")
  # code for prediction 
    cost = ""
    
    if st.button("Test Results : "):
        cost = health_insurance_cost([Age,Sex,bmi,children,smoker])
    
    st.success(cost)

if __name__ == "__main__":
    main()
    
