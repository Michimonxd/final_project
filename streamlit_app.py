import pandas as pd 
from sklearn import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import streamlit as st
import pandas as pd
import joblib
import time
import json
import streamlit.components.v1 as components


# Initialize LabelEncoder
label_encoder = LabelEncoder()
        
# Manually defined label mapping for predictions
label_mapping = {
    0: 'ARP_poisioning',
    1: 'DDOS_Slowloris',
    2: 'DOS_SYN_Hping',
    3: 'MQTT_Publish',
    4: 'Metasploit_Brute_Force_SSH',
    5: 'NMAP_FIN_SCAN',
    6: 'NMAP_OS_DETECTION',
    7: 'NMAP_TCP_scan',
    8: 'NMAP_UDP_SCAN',
    9: 'NMAP_XMAS_TREE_SCAN',
    10: 'Thing_Speak',
    11: 'Wipro_bulb'
}


# Load the pre-trained model and scaler
rf_model = joblib.load('random_forest_model_smote.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')



# Function to display an alert if a specific attack type is detected
def check_attack_type(prediction):
    """
    Display an alert if the given prediction is one of the following attack types:
        - DDOS_Slowloris
        - NMAP_TCP_scan
        - ARP_poisioning
        - DOS_SYN_Hping
        - NMAP_UDP_SCAN
        - NMAP_XMAS_TREE_SCAN
        - NMAP_OS_DETECTION
        - NMAP_FIN_SCAN
        - NMAP_TCP_scan
        - Metasploit_Brute_Force_SSH
    
    The alert is displayed as a red box with white text and a blinking animation.
    """
    alert_attack_types = ['DDOS_Slowloris', 'NMAP_TCP_scan', 'ARP_poisioning', 'DOS_SYN_Hping', 'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN', 'NMAP_OS_DETECTION', 'NMAP_FIN_SCAN', 'NMAP_TCP_scan', 'Metasploit_Brute_Force_SSH', ]  
    
    if prediction in alert_attack_types:
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 5px; background-color: red; color: white; text-align: center; animation: blinker 1s linear infinite;">
                <strong>ALERT!</strong> {prediction} detected!
            </div>
            <style>
            @keyframes blinker {{
                50% {{ opacity: 0; }}
            }}
            </style>
            """,
            unsafe_allow_html=True)


# Main function for the Streamlit app
def main():
    st.sidebar.title("Real-Time Data Simulation")
    
    # Upload a CSV file in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    # Initialize session state for storing results
    if 'running' not in st.session_state:
        st.session_state['running'] = False
    if 'results' not in st.session_state:
        st.session_state['results'] = []

    # Create start/stop buttons
    start_button = st.sidebar.button("Start Simulation")
    stop_button = st.sidebar.button("Stop Simulation")

    if start_button:
        st.session_state['running'] = True
        st.session_state['results'] = []  

    if stop_button:
        st.session_state['running'] = False

    # Check if a file has been uploaded
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")

        # Apply LabelEncoder to each categorical column
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                df[column] = label_encoder.fit_transform(df[column])

        # Align columns to match the training data
        df_aligned = align_columns(df, feature_names)

        # Tabs for data preview and results
        tab1, tab2 = st.tabs(["Data Preview", "Prediction Results"])

        with tab1:
            st.write("### Data Preview")
            st.dataframe(df.head())  
            st.write('### Aligned Data')
            st.dataframe(df_aligned.head())

    with tab2:
        if st.session_state['running']:
            with st.spinner("Running real-time simulation..."):
                simulate_real_time_feed_from_df(df_aligned)
        else:
            if st.session_state['results']:
                st.write("### Previous Prediction Results")
                for result in st.session_state['results']:
                    # Display the prediction result with smaller font size
                    st.markdown(
                        f"<p style='font-size:14px;'><strong>Top Predicted Attack Type:</strong> "
                        f"<span style='font-size:12px;'>{result['label']}</span> "
                        f"({result['probability']:.2%})</p>",
                        unsafe_allow_html=True
                    )
                    with st.expander("Feature Values"):
                        st.write(result['data'])
                        
                        
# Function to shuffle and simulate real-time data feeding and trigger alerts
def simulate_real_time_feed_from_df(df, delay=1):
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total_rows = len(shuffled_df)
    progress_bar = st.progress(0)

    for i, row in shuffled_df.iterrows():
        if not st.session_state['running']:
            break  

        df_row = pd.DataFrame([row], columns=feature_names)
        new_data_scaled = scaler.transform(df_row.values)
        new_data_scaled_df = pd.DataFrame(new_data_scaled, columns=feature_names)

        prediction = rf_model.predict(new_data_scaled_df)
        prediction_proba = rf_model.predict_proba(new_data_scaled_df)[0]

        prediction_label = label_mapping.get(prediction[0], "Unknown")
        top_prediction = prediction_proba.max()
        top_index = prediction_proba.argmax()
        top_label = label_mapping.get(top_index, "Unknown")

        # Store results in session state
        st.session_state['results'].append({
            'label': top_label,
            'probability': top_prediction,
            'data': df_row
        })

        # Display result
        st.metric(label="Top Predicted Attack Type", value=top_label, delta=f"{top_prediction:.2%}")
        with st.expander("Feature Values"):
            st.write(df_row)

        # Trigger an alert if a specific attack type is detected
        check_attack_type(prediction_label)

        # Update progress bar
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)

        time.sleep(delay)

    if not st.session_state['running']:
        st.warning("Simulation stopped.")

# Utility function to align columns
def align_columns(df, feature_names):
    aligned_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in df.columns:
            aligned_df[col] = df[col]
        else:
            aligned_df[col] = 0  
    aligned_df = aligned_df.reindex(columns=feature_names, fill_value=0)  
    return aligned_df

main()