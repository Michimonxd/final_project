# Code to create and save 'streamlit_app.py' from Jupyter Notebook
code = """
import numpy as np
import joblib  # To load the trained model and scaler
import streamlit as st

# Load the trained model and scaler
model = joblib.load('rf_smote_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("IoT Security Monitoring")
st.write("Enter the feature values as comma-separated numbers:")

# Input for feature values
features = st.text_input("Feature values (comma-separated):")

if st.button("Send IoT Data"):
    try:
        # Convert input string to a NumPy array
        features_array = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
        
        # Normalize the input features using the loaded scaler
        features_scaled = scaler.transform(features_array)
        
        # Make predictions
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)

        # Display the prediction status
        status = "Attack Detected" if prediction[0] else "No Attack"
        st.write(f"Status: {status}")
        st.write(f"Confidence: Class 0: {prediction_proba[0][0]:.2f}, Class 1: {prediction_proba[0][1]:.2f}")
    except ValueError:
        st.error("Please enter valid numeric feature values.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
"""

# Write the code to a Python file
with open('streamlit_app.py', 'w') as file:
    file.write(code)

print("File 'streamlit_app.py' has been saved successfully.")

