import joblib
import streamlit as st
import numpy as np
import pandas as pd
model=joblib.load("superstore_best_rs_rf_model.pkl")
Sales=st.number_input('Sales', min_value=0, max_value=22700, value="0", step=1)
Discount=st.number_input('Discount', min_value=0, max_value=100, value="0", step=1)
Shipping_cost=st.number_input('Shipping', min_value=0, max_value=100, value="0", step=.001, format="%.3f")
if st.button("Predict HDB price"):

    ## Create dict for input features
    input_data = {
        'Sales': Sales,
        'Discount': Discount,
        'Shipping': Shipping_cost,
    }
    
    df_input = pd.DataFrame({
        'Sales': Sales,
        'Discount': Discount,
        'Shipping': Shipping_cost,
    })
    df_input = pd.get_dummies(df_input, 
                              columns = ['Sales', 'Discount', 'Shipping']
                             )
    

    df_input = df_input.reindex(columns = model.feature_names_in_,
                                fill_value=0)



    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Resale Price: ${y_unseen_pred:,.2f}")
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://www.shutterstock.com/shutterstock/videos/1025418011/thumb/1.jpg");
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
)