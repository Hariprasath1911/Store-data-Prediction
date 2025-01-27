import streamlit as st
from datetime import date
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ReLU
import base64
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: fit;
            background-repeat: repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local(r"12.png")
@st.cache_resource
def load_model_file():
    # Handle custom layers (if applicable)
    custom_objects = {"ReLU": ReLU}
    return load_model("model_store.h5", custom_objects=custom_objects)

# Load the model
model = load_model_file()

with open("std_store.pkl","rb") as f:
    scaler=pickle.load(f)

st.title("Weekly Sales Forecast for Departmental Store: Unlocking Insights for Better Planning")

st.subheader("Basic Store Data")
Store = st.number_input('Enter Store number(1-45)', min_value=1)
Dept = st.number_input('Enter Department number(1-99)',min_value=1)
Size= st.number_input('Enter Squarefeet of your store',min_value=0.1)
Type_select=st.selectbox('Select Job Type',['A','B','C'])
Type_map={'B':1,'A':0,'C':2}
Type=Type_map.get(Type_select)

st.subheader("Store Related Data")
selected_date = st.date_input(
    "Choose a date:",
    value=date.today(), 
    min_value=date(2000, 1, 1),
    max_value=date(2040, 12, 31)
)
if selected_date:
    selected_datetime = datetime.combine(selected_date, datetime.min.time())
    
    Year = selected_datetime.year
    Month = selected_datetime.month
    Day = selected_datetime.day
    Day_of_week = selected_datetime.weekday()
    week = selected_datetime.isocalendar()[1]

holi=st.selectbox('Is there any Public Holiday in the particular week',['Yes','No'])
holi_map={'Yes':1,'No':0}    
IsHoliday=holi_map.get(holi)
Temperature=st.number_input('Enter temperature',min_value=0.01)
Fuel_Price=st.number_input('Enter Fuel Price',min_value=0.01)
MarkDown1=st.number_input('Enter MarkDown1',min_value=0)
MarkDown2=st.number_input('Enter MarkDown2',min_value=0)
MarkDown3=st.number_input('Enter MarkDown3',min_value=0)
MarkDown4=st.number_input('Enter MarkDown4',min_value=0)
MarkDown5=st.number_input('Enter MarkDown5',min_value=0)
CPI=st.number_input('Enter CPI',min_value=0.0001)
Unemployment1=st.number_input('Enter Unemployment rate',min_value=0.001)
Unemployment=np.log2(Unemployment1)

if st.button('Predict'):
    c1=np.array([[Store,Dept,IsHoliday,Type,Size,Temperature,Fuel_Price,MarkDown1,MarkDown2,MarkDown3,MarkDown4,MarkDown5,CPI,Unemployment,Year,Month,Day,Day_of_week,week]])
    c2=scaler.transform(c1)
    predicted_output = model.predict(c2)
    st.write(f"Predicted Value: {predicted_output[0][0]:.2f}")
