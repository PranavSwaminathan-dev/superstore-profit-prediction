import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Superstore Profit Predictor", page_icon="💰", layout="centered")

model = joblib.load("superstore_blend_model.pkl")




st.title("💰 Superstore Profit Predictor")
st.write(
    "Estimate the profit of an order based on its sales, discount, shipping "
    "cost, quantity, and product sub-category — powered by a Random Forest "
    "model trained on Global Superstore order data."
)

st.subheader("Order Details")

col1, col2 = st.columns(2)

with col1:
    sales = st.number_input(
        "Sales ($)", min_value=0.0, max_value=22700.0, value=100.0, step=10.0
    )
    discount = st.slider(
        "Discount", min_value=0.0, max_value=1.0, value=0.0, step=0.01
    )
    quantity = st.slider("Quantity", min_value=1, max_value=14, value=1)

with col2:
    shipping_cost = st.number_input(
        "Shipping Cost ($)", min_value=0.0, max_value=1000.0, value=10.0, step=1.0
    )
    sub_category = st.selectbox(
        "Sub-Category",
        [
            "Accessories", "Appliances", "Art", "Binders", "Bookcases",
            "Chairs", "Copiers", "Envelopes", "Fasteners", "Furnishings",
            "Labels", "Machines", "Paper", "Phones", "Storage",
            "Supplies", "Tables",
        ],
    )

predict_clicked = st.button("Predict Profit", type="primary")

if predict_clicked:
    # Derive the engineered features the model was trained on, from live inputs
    unit_price = sales / quantity
    shipping_ratio = shipping_cost / sales if sales > 0 else 0.0

    input_dict = {
        "Sales": sales,
        "Discount": discount,
        "Shipping Cost": shipping_cost,
        "Quantity": quantity,
        "Unit_Price": unit_price,
        "Shipping_Ratio": shipping_ratio,
    }
    df_input = pd.DataFrame([input_dict])

    # One-hot flag for the selected sub-category (matches training's pd.get_dummies)
    df_input[f"Sub-Category_{sub_category}"] = 1

    # Align to the exact columns the model was trained on; anything not set
    # (other sub-categories, countries, segments, etc.) defaults to 0
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

    try:
        prediction = model.predict(df_input)[0]
        st.success(f"### Predicted Profit: ${prediction:,.2f}")
        if prediction < 0:
            st.warning(
                "This order is predicted to be a loss. Try lowering the "
                "discount or shipping cost to see the impact."
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Fill in the order details above and click **Predict Profit** to see the estimate.")