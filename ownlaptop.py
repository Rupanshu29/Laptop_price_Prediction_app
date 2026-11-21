import pandas as pd
import numpy as np
import streamlit as st
import joblib

lr = joblib.load("C:\\ML\\own_done_laptop_lr.joblib")
one = joblib.load("C:\\ML\\own_done_laptop_one.joblib")

st.title("Laptop Price Prediction app")

col1, col2, col3 = st.columns([1,1,1])

sb = col1.selectbox("Select_Brand",["Acer","Asus","Dell","HP","Lenovo"])

ps = col2.number_input("Processor_Speed",1.5,3.99,1.5,0.1)

rs = col3.selectbox("RAM_size",[4,8,16,32])

sc = col1.selectbox("Storage_Capacity",[256,512,1000])

ss = col2.number_input("Screen_Size",11.01,16.98,11.01,0.1)

w = col3.number_input("Weight",2.00,4.99,2.00,0.10)

prd = pd.DataFrame({"Brand":[sb],
                    "Processor_Speed":[ps],
                    "RAM_Size":[rs],
                    "Storage_Capacity":[sc],
                    "Screen_Size":[ss],
                    "Weight":[w]})

st.write(prd)
prd_cat = prd[["Brand"]]
prd_num = prd.drop(columns=["Brand"])

# Encode Brand
prd_data_one = one.transform(prd_cat)

# Create encoded DataFrame
prd_data_one = pd.DataFrame(
    prd_data_one,
    columns=one.get_feature_names_out(["Brand"]),
    index=prd.index
)

# Combine numerical and encoded data
prd_final = pd.concat([prd_num, prd_data_one], axis=1)

# Predict price
b1 = st.button("Predict Price")

if b1:
    price = lr.predict(prd_final)
    st.title(f"₹{round(price[0],0)}")

