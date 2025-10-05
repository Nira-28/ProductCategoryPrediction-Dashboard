import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("Predict Product Category")

# --- Load Dataset ---
dataset_path = "Online Sales Data.csv"  # Replace with your dataset path
try:
    # Try default comma separator
    df = pd.read_csv(dataset_path)
except:
    # Fallback: try semicolon
    df = pd.read_csv(dataset_path, sep=';')

st.write("Columns in dataset:", df.columns)
st.write("First few rows:", df.head())

# --- Data Cleaning ---
# Replace infinite values and drop rows with NaN
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# --- Encode categorical columns ---
region_encoder = LabelEncoder()
df['region_encoded'] = region_encoder.fit_transform(df['Region'])

payment_encoder = LabelEncoder()
df['payment_encoded'] = payment_encoder.fit_transform(df['Payment Method'])

product_label_encoder = LabelEncoder()
df['product_encoded'] = product_label_encoder.fit_transform(df['Product Category'])

# --- Train Model ---
features = ['Units Sold', 'Unit Price', 'Total Revenue', 'region_encoded', 'payment_encoded']
X = df[features]
y = df['product_encoded']

model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders for reuse
with open("product_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
with open("region_encoder.pkl", "wb") as f:
    pickle.dump(region_encoder, f)
with open("payment_encoder.pkl", "wb") as f:
    pickle.dump(payment_encoder, f)
with open("product_label_encoder.pkl", "wb") as f:
    pickle.dump(product_label_encoder, f)

st.success("Model trained and encoders saved!")

# --- Dashboard Inputs ---
st.header("Predict a Product Category")

units_sold = st.number_input("Units Sold", min_value=1, step=1)
unit_price = st.number_input("Unit Price", min_value=0.0, step=0.1)
total_revenue = units_sold * unit_price
st.markdown(f"**Total Revenue: {total_revenue:.2f}**")

region_options = region_encoder.classes_.tolist()
payment_options = payment_encoder.classes_.tolist()

region_input = st.selectbox("Region", region_options)
payment_input = st.selectbox("Payment Method", payment_options)

# Encode inputs
region_encoded = region_encoder.transform([region_input])[0]
payment_encoded = payment_encoder.transform([payment_input])[0]

features_input = np.array([[units_sold, unit_price, total_revenue, region_encoded, payment_encoded]])

# Predict
if st.button("Predict Product Category"):
    prediction = model.predict(features_input)
    predicted_label = product_label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Product Category: **{predicted_label}**")
