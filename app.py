import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
model_path = "V:\\Kenyan data\\kenya_tourism_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the prediction function
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# Create a Streamlit app
def main():
    st.title("Kenya Tourism Prediction App")
    st.write("Enter the details below to predict total cost:")
    
    # Create input fields for user input
    total_female = st.number_input("Total Female")
    total_male = st.number_input("Total Male")
    nights_stayed = st.number_input("Number of Nights Stayed")
    total_people = total_female + total_male
    
    # Convert user input into a DataFrame
    input_data = pd.DataFrame({
        'total_female': [total_female],
        'total_male': [total_male],
        'nights_stayed': [nights_stayed],
        'total_people': [total_people]
    })
    
    # Make predictions when the user clicks the predict button
    if st.button("Predict"):
        prediction = predict(input_data)
        st.success(f"The predicted total cost is {prediction}")

if __name__ == "__main__":
    main()
