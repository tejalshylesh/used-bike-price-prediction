import os
import subprocess
import sys
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MotoValue | Used Bike Price Predictor", page_icon="🏍️", layout="centered")

@st.cache_resource
def load_bundle():
    if not os.path.exists("bike_price_model.pkl"):
        subprocess.run(
            [sys.executable, "train_model.py"],
            check=True
        )
    return joblib.load("bike_price_model.pkl")

st.markdown("""
<style>
.block-container {max-width: 850px; padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#142d2a,#234f45); padding: 1.6rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin-bottom:.25rem;}
.hero p {color:#d4e7df; margin-bottom:0;}
</style>
<div class="hero"><h1>🏍️ MotoValue</h1><p>USED BIKE PRICE PREDICTOR · Machine Learning</p><h3>Know what your bike is worth.</h3><p>Enter your bike details to estimate its resale value.</p></div>
""", unsafe_allow_html=True)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("The trained model was not found. First run: python train_model.py")
    st.stop()

st.caption(f"Prediction model: {bundle['model_name']}")
st.subheader("Bike details")
st.write("Select the available details below, then predict the estimated resale price.")
inputs = {}
data = bundle["training_data"]

with st.form("bike_prediction_form"):
    left, right = st.columns(2)
    for i, feature in enumerate(bundle["features"]):
        target_col = left if i % 2 == 0 else right
        label = {"bike_name":"Bike model/name", "city":"City", "kms_driven":"Kilometres driven", "owner":"Owner category", "age":"Bike age", "power":"Power / CC", "brand":"Brand"}.get(feature, feature.replace("_", " ").title())
        with target_col:
            if feature in bundle["categorical"]:
                options = sorted(data[feature].dropna().astype(str).unique().tolist(), key=str.lower)
                if not options:
                    options = ["Unknown"]
                inputs[feature] = st.selectbox(label, options)
            else:
                vals = pd.to_numeric(data[feature], errors="coerce").dropna()
                minimum = float(max(0, vals.min()))
                maximum = float(vals.max())
                if maximum <= minimum:
                    maximum = minimum + 1.0
                default = float(vals.median())
                default = min(max(default, minimum), maximum)
                step = 1.0
                inputs[feature] = st.number_input(label, min_value=minimum, max_value=maximum, value=default, step=step)
    submitted = st.form_submit_button("Predict Bike Price", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([inputs], columns=bundle["features"])
    prediction = max(0.0, float(bundle["model"].predict(row)[0]))
    st.markdown("---")
    st.success(f"### Estimated resale price: ₹{prediction:,.0f}")
    st.caption("This is an estimate based on historical listings, not a guaranteed selling price.")

with st.expander("View model evaluation"):
    metrics = pd.DataFrame(bundle["results"]).T
    st.dataframe(metrics.style.format({"MAE":"₹{:,.2f}", "RMSE":"₹{:,.2f}", "R2":"{:.4f}"}), use_container_width=True)
    st.write(f"Dataset records used: {bundle['dataset_rows']:,}")
    st.caption("Lower MAE/RMSE is generally better. Interpret these metrics together.")

st.markdown("<p style='text-align:center;color:gray;font-size:12px'>IDS Mini Project · Used Bike Price Prediction</p>", unsafe_allow_html=True)
