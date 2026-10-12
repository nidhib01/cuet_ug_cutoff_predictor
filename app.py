import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configure page
st.set_page_config(page_title="CUET UG Cutoff Predictor", layout="centered")

st.title("🎯 CUET UG Cutoff Predictor")
st.write("Enter the details below to predict the target reservation category cutoff.")

@st.cache_resource
def load_assets():
    # Load the model bundle
    bundle = joblib.load('model.pkl')
    # Load the standalone encoder
    encoder = joblib.load('scaler.pkl')
    return bundle['model'], encoder, bundle['encoder_y'], bundle['feature_cols']

try:
    model, encoder_X, encoder_y, feature_cols = load_assets()
    
    # Get category classes
    categories = list(encoder_y.classes_)
    
    # Create form fields dynamically from encoder categories
    st.subheader("Predictor Form")
    
    # Retrieve unique values per feature column from the encoder
    unis = list(encoder_X.categories_[0])
    years = [int(y) for y in encoder_X.categories_[1]]
    rounds = list(encoder_X.categories_[2])
    seat_types = list(encoder_X.categories_[3])
    
    selected_uni = st.selectbox("University", options=unis)
    selected_year = st.selectbox("Year", options=years)
    selected_round = st.selectbox("Admission Round", options=rounds)
    selected_seat_type = st.selectbox("Seat Type", options=seat_types)
    
    if st.button("Predict Category Cutoff"):
        # Construct input DataFrame
        input_df = pd.DataFrame([[
            selected_uni, 
            selected_year, 
            selected_round, 
            selected_seat_type
        ]], columns=feature_cols)
        
        # Encode the inputs using our saved encoder
        encoded_inputs = encoder_X.transform(input_df)
        
        # Predict and decode
        pred_encoded = model.predict(encoded_inputs)
        prediction = encoder_y.inverse_transform(pred_encoded)[0]
        
        st.success(f"Predicted Target Category Cutoff: **{prediction}**")

except Exception as e:
    st.error(f"Error initializing application: {e}")
    st.info("Make sure 'model.pkl' and 'scaler.pkl' are in the same directory as 'app.py'")