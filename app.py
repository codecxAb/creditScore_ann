import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random

# Load the trained model
try:
    model = tf.keras.models.load_model("credit_model_improved.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the original dataframe
try:
    df_original = pd.read_csv('estadistical.csv')
except FileNotFoundError:
    st.error("Error: 'estadistical.csv' not found.")
    st.stop()

# Identify categorical and numerical columns
categorical_cols_original = df_original.select_dtypes(include=['object']).columns.tolist()
numerical_cols_original = df_original.select_dtypes(include=np.number).columns.tolist()
if 'Receive/ Not receive credit ' in numerical_cols_original:
    numerical_cols_original.remove('Receive/ Not receive credit ')

# Initialize LabelEncoders and fit them on the original data
label_encoders = {}
for col in categorical_cols_original:
    le = LabelEncoder()
    df_original[col] = le.fit_transform(df_original[col])
    label_encoders[col] = le

# Initialize StandardScaler and fit it on the numerical features of the original data
scaler = StandardScaler()
X_original = df_original.drop('Receive/ Not receive credit ', axis=1)
scaler.fit(X_original[numerical_cols_original])
feature_order = X_original.columns.tolist()

st.title("Simplified Credit Approval Prediction")
st.write("Enter the following details:")

input_data = {}

# Get user inputs for the specified features
input_data['Duration in month'] = st.number_input("Enter Duration in months", value=12, step=1)
input_data['Credit amount'] = st.number_input("Enter Credit Amount", value=2500.0, step=100.0)
input_data['Age in years'] = st.number_input("Enter Age", value=30, step=1)
input_data['Present residence since'] = st.number_input("Enter Years at Current Residence", value=2, step=1)
input_data['Number of existing credits at this bank'] = st.number_input("Enter Number of Existing Credits", value=1, step=1)
input_data['Number of people being liable to provide maintenance for'] = st.number_input("Enter Number of Dependents", value=1, step=1)

if st.button("Predict"):
    input_data_processed = {}
    for col in feature_order:
        if col in input_data:
            input_data_processed[col] = input_data[col]
        else:
            if col in categorical_cols_original:
                input_data_processed[col] = random.choice(list(label_encoders[col].classes_))
            elif col in numerical_cols_original:
                input_data_processed[col] = df_original[col].median()

    # Create a DataFrame from the processed input
    input_df = pd.DataFrame([input_data_processed])[feature_order]

    # Separate numerical and categorical features for scaling and encoding
    numerical_input = input_df[numerical_cols_original]
    categorical_input = input_df[categorical_cols_original]

    # Scale numerical features
    scaled_numerical_input = scaler.transform(numerical_input)
    scaled_df = pd.DataFrame(scaled_numerical_input, columns=numerical_cols_original, index=input_df.index)

    # Encode categorical features
    encoded_categorical_input = pd.DataFrame()
    for col in categorical_input.columns:
        if col in label_encoders:
            encoded_categorical_input[col] = label_encoders[col].transform(categorical_input[col].astype(str))
        else:
            encoded_categorical_input[col] = categorical_input[col]

    # Combine the processed numerical and categorical features
    final_input_df = pd.concat([scaled_df, encoded_categorical_input], axis=1)
    final_input_df = final_input_df[feature_order] # Ensure correct column order

    # Predict
    prediction_probability = model.predict(final_input_df)[0][0]
    prediction = "✅ Will Receive Credit" if prediction_probability >= 0.5 else "❌ Will Not Receive Credit"
    confidence = f"{prediction_probability * 100:.2f}%"

    st.subheader("Prediction Result:")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence}")