import streamlit as st
import pickle

# Load model
model = pickle.load(open('hotel_model.pkl', 'rb'))

# App Title
st.title("🏨 Expedia Hotel Recommender")
st.write("Fill in the details to get a hotel cluster recommendation:")

# User inputs
country = st.number_input("User Location Country", min_value=1, value=66)
destination = st.number_input("Search Destination ID", min_value=1, value=8250)
is_mobile = st.selectbox("Is Mobile?", [0, 1])
distance = st.number_input("Original Destination Distance", min_value=0.0, value=1200.0)

# Predict button
if st.button("Recommend Hotel"):
    input_data = [[country, destination, is_mobile, distance]]
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Hotel Cluster: {prediction[0]}")
