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

#Load Dataframe
df = pd.read_csv('IOT_cleaned.csv')

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to each categorical column
for column in df.columns:
    if df[column].dtype == 'object' or df[column].dtype.name == 'category':
        df[column] = label_encoder.fit_transform(df[column]) 
        
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

features = df.drop(columns=['Attack_type', 'Attack'])
target = df["Attack_type"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state=0)

#Normalize Data after Train Split
normalizer = MinMaxScaler() #define normalizer

normalizer.fit(X_train)

X_train_norm = normalizer.transform(X_train) # Normalize 80% training Data
X_test_norm = normalizer.transform(X_test) # Normalize 20% Testing Data

#Apply to test and training data
X_train_norm = pd.DataFrame(X_train_norm, columns = X_train.columns)
X_test_norm = pd.DataFrame(X_test_norm, columns = X_test.columns)

from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=0)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Apply SelectKBest to the resampled (SMOTE) training data
selector = SelectKBest(score_func=f_classif, k=50)  
X_kbest_smote = selector.fit_transform(X_smote, y_smote)  # Use the resampled data (X_smote, y_smote)

# Transform the original test set using the fitted selector
X_test_selected = selector.transform(X_test)  # Apply the transformation to the test set

# Get selected feature names
selected_features = X_smote.columns[selector.get_support()]  # Ensure columns are from the training data
print("Selected features:", selected_features)

# Apply MinMaxScaler 
scaler = MinMaxScaler()
X_kbest_smote_scaled = scaler.fit_transform(X_kbest_smote)  # Scale the resampled and selected training data
X_test_selected_scaled = scaler.transform(X_test_selected)  # Scale the transformed test set

# Convert scaled arrays back to DataFrames for better readability
X_kbest_smote_scaled = pd.DataFrame(X_kbest_smote_scaled, columns=selected_features)
X_test_selected_scaled_smote = pd.DataFrame(X_test_selected_scaled, columns=selected_features)

def get_column_names(X_test_selected_scaled_smote):
    # Get the column names from the DataFrame or array
    if hasattr(X_test_selected_scaled_smote, 'columns'):
        # If it's a DataFrame, return column names
        return X_test_selected_scaled_smote.columns.tolist()
    else:
        # If it's a numpy array, you may need to access column names if available
        return ["Column_" + str(i) for i in range(X_test_selected_scaled_smote.shape[1])]

# Get column names from the DataFrame or array
column_names = get_column_names(X_test_selected_scaled_smote)

# Convert the column names list into a DataFrame (with one row to view)
column_names_df = pd.DataFrame([column_names], columns=column_names)

# Load the pre-trained model and scaler
rf_model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

# Main function for the Streamlit app
def main():
    st.title("Real-Time Data Simulation and Prediction App")

    # Upload a CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.dataframe(df)

        # Display extracted column names
        column_names = get_column_names(df)
        st.write("Extracted Column Names:")
        st.write(column_names)

        # Save the column names as a DataFrame for use in the function
        column_names_df = pd.DataFrame(columns=column_names)

        # Fit the MinMaxScaler on the DataFrame (excluding target columns)
        scaler.fit(df.drop(['Attack_type', 'Attack'], axis=1, errors='ignore'))

        # Run the real-time data simulation function
        if st.button("Start Real-Time Simulation"):
            simulate_real_time_feed_from_df(df)

# Function to shuffle and simulate real-time data feeding with row and probability printing
def simulate_real_time_feed_from_df(df, delay=1):
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for _, row in shuffled_df.iterrows():
        new_data_row = row.drop(['Attack_type', 'Attack'], errors='ignore')
        new_data_row = pd.DataFrame([new_data_row], columns=column_names_df.columns)
        
        label_encoder = LabelEncoder()
        for column in new_data_row.columns:
            if new_data_row[column].dtype == 'object' or new_data_row[column].dtype.name == 'category':
                new_data_row[column] = label_encoder.fit_transform(new_data_row[column])

        st.write("\nFeature Values (Raw):")
        st.write(new_data_row)

        new_data_scaled = scaler.transform(new_data_row)
        new_data_scaled_df = pd.DataFrame(new_data_scaled, columns=column_names_df.columns)

        prediction = rf_model.predict(new_data_scaled_df)
        prediction_proba = rf_model.predict_proba(new_data_scaled_df)

        prediction_label = label_mapping.get(prediction[0], "Unknown")

        st.write(f"\nPredicted Attack Type: {prediction_label}")
        st.write(f"Prediction Probabilities: {prediction_proba[0]}")

        time.sleep(delay)

if __name__ == "__main__":
    main()
