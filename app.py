import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
@st.cache_data
def load_model_and_scaler():
    model_path = 'isolation_forest_model.pkl'
    scaler_path = 'scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_scaler()

def preprocess_data(data, non_numeric_cols, scaler, original_features):
    # Convert categorical columns to numeric using one-hot encoding
    data_encoded = pd.get_dummies(data, columns=non_numeric_cols, drop_first=True)
    
    # Ensure data has the same columns as the original features
    for col in original_features:
        if col not in data_encoded.columns:
            data_encoded[col] = 0  # Add missing columns with default values
            
    data_encoded = data_encoded[original_features]  # Reorder columns to match original data
    
    # Handle missing values
    data_encoded = data_encoded.fillna(data_encoded.mean())
    
    # Scale the features
    data_scaled = scaler.transform(data_encoded)
    
    return data_scaled

def detect_fraudulent_transactions(new_data, model, scaler):
    # Load the original training data to get feature names
    original_data_path = 'AssignmentData.xlsx'  # Adjust this path
    original_data = pd.read_excel(original_data_path, sheet_name='creditcard_test')
    
    # Identify non-numeric columns and encode the training data
    non_numeric_cols = original_data.select_dtypes(include=['object']).columns
    original_data_encoded = pd.get_dummies(original_data, columns=non_numeric_cols, drop_first=True)
    
    # Extract the feature names
    original_features = original_data_encoded.columns
    
    # Preprocess new data
    new_data_scaled = preprocess_data(new_data, non_numeric_cols, scaler, original_features)
    
    # Predict anomalies
    predictions = model.predict(new_data_scaled)
    predictions = [1 if p == -1 else 0 for p in predictions]
    
    # Add predictions to the new data
    new_data['Prediction'] = predictions
    
    # Filter fraudulent transactions
    fraudulent_transactions = new_data[new_data['Prediction'] == 1]
    
    return fraudulent_transactions

# Streamlit app
st.title('Credit Card Fraud Detection')
uploaded_file = st.file_uploader("Upload a new set of credit card transactions:", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load the dataset
        new_data = pd.read_excel(uploaded_file, sheet_name='creditcard_test')
        
        # Detect fraudulent transactions
        fraudulent_transactions = detect_fraudulent_transactions(new_data, model, scaler)
        
        # Display results
        st.write("Fraudulent Transactions:")
        st.dataframe(fraudulent_transactions)
    except Exception as e:
        st.error(f"An error occurred: {e}")
