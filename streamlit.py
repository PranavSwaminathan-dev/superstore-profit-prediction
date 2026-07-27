import joblib
import streamlit as st
import numpy as np
import pandas as pd

model = joblib.load("superstore_best_rf_model.pkl")

st.title("Superstore Profit Predictor")

Sales = st.number_input('Sales', min_value=0, max_value=22700, value=100, step=1)
Discount = st.number_input('Discount', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
Shipping_Cost = st.number_input('Shipping Cost', min_value=0.0, max_value=1000.0, value=10.0, step=0.01, format="%.2f")
Quantity= st.slider('Quantity',min_value=1,max_value=14,step=1,value=1)
Year=st.number_input("Year", min_value=2011, max_value=2026, value=2014, step=1)
if st.button("Predict Profit"):

    ## Build a single-row DataFrame — values wrapped in lists, not passed as scalars
    df_input = pd.DataFrame({
        'Sales': [Sales],
        'Discount': [Discount],
        'Shipping Cost': [Shipping_Cost], 
        'Quantity': [Quantity],
        'Year':[Year],
    })


    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    y_pred = model.predict(df_input)[0]
    st.success(f"Predicted Profit: ${y_pred:,.2f}")