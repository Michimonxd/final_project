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
import altair as alt

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
    """
    Main entry point for the Streamlit app.
    """
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

    # Create tabs for Data Preview, Prediction Results, and Summary
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Prediction Results", "Summary"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")

        # Apply LabelEncoder to each categorical column
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype.name == 'category':
                df[column] = label_encoder.fit_transform(df[column])

        # Align columns to match the training data
        df_aligned = align_columns(df, feature_names)

        with tab1:
            st.write("### Uploaded Data")
            st.dataframe(df.head())  
            st.write('### Transformed Data')
            st.dataframe(df_aligned.head())

        with tab2:
            if st.session_state['running']:
                with st.spinner("Running real-time simulation..."):
                    simulate_real_time_feed_from_df(df_aligned)
            else:
                if st.session_state['results']:
                    st.write("### Previous Prediction Results")
                    for result in st.session_state['results']:
                        st.markdown(
                            f"<p style='font-size:14px;'><strong>Top Predicted Attack Type:</strong> "
                            f"<span style='font-size:12px;'>{result['label']}</span> "
                            f"({result['probability']:.2%})</p>",
                            unsafe_allow_html=True
                        )
                        with st.expander("Feature Values"):
                            st.write(result['data'])

        with tab3:
            st.write("### Prediction Counts")
            if st.session_state['results']:
                # Count of each prediction label
                prediction_labels = [result['label'] for result in st.session_state['results']]
                prediction_counts = pd.Series(prediction_labels).value_counts().reset_index()
                prediction_counts.columns = ['Prediction', 'Count']

                # color bar chart
                prediction_counts['Color'] = prediction_counts['Prediction'].apply(
                    lambda x: 'blue' if x in ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb'] else 'red'
                )

            
                bar_chart = alt.Chart(prediction_counts).mark_bar().encode(
                    x=alt.X('Prediction:N', sort='-y'),
                    y='Count:Q',
                    color=alt.Color('Color:N', scale=None),  
                    tooltip=['Prediction', 'Count']
                ).properties(
                    width=600,
                    height=400
                )

                #display bar chart
                st.altair_chart(bar_chart, use_container_width=True)

    else:
        with tab1:
            st.write("Please upload a CSV file to preview data.")

        with tab2:
            st.write("No results to display yet. Upload a file and start the simulation.")

        with tab3:
            st.write("No summary available yet. Start a simulation to view prediction counts.")


                        
                        
# Function to shuffle and simulate real-time data feeding and trigger alerts
def simulate_real_time_feed_from_df(df, delay=1):
    """
    Simulate real-time data feeding and prediction using the given DataFrame.

    This function:
    1. Shuffles the given DataFrame to randomize row order.
    2. Iterates over each row in the shuffled DataFrame.
    3. Applies scaling to the row using the given scaler.
    4. Gets the prediction and probabilities from the given model.
    5. Stores the result in the session state.
    6. Displays the result as a metric with a delta showing the top predicted probability.
    7. Triggers an alert if a specific attack type is detected.
    8. Updates a progress bar to show the progress of the simulation.
    9. Sleeps for the given delay between each row processing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to simulate.
    delay : int, optional
        Delay in seconds between each row of the DataFrame being processed,
        by default 1.
    """
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

        # store result so it doesnt disappear
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
    """
    Align a DataFrame's columns with the given feature names.

    This function takes a DataFrame and a list of feature names and returns a new
    DataFrame with the same data, but with columns aligned to match the feature
    names. If a column is present in the original DataFrame but not in the feature
    names, it is omitted from the aligned DataFrame. If a column is present in the
    feature names but not in the original DataFrame, it is added to the aligned
    DataFrame with a value of 0.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to align.
    feature_names : list
        The list of feature names to align the DataFrame to.

    Returns
    -------
    pandas.DataFrame
        The aligned DataFrame.
    """
    aligned_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in df.columns:
            aligned_df[col] = df[col]
        else:
            aligned_df[col] = 0  
    aligned_df = aligned_df.reindex(columns=feature_names, fill_value=0)  
    return aligned_df

main()