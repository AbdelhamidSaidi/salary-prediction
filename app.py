import streamlit as st
import pandas as pd
import joblib

st.title("Salary Prediction App")
st.divider()

st.write("with this app you can get an estimate of the salary based on the years of experience")

years = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)
jobrate = st.number_input("Job Rate", min_value=0.0, max_value=500.0, step=0.1)

X= [[years, jobrate]]

model = joblib.load('salary_model.pkl')

st.divider()

predict = st.button("Predict Salary")
if predict:
    salary = model.predict(X)
    st.success(f"The estimated salary is: ${salary[0]:,.2f} per year")