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



# Data integrity check function for feature-selected data
def check_data_integrity(X_train, X_test, y_train, y_test, name):
    print(f"\n--- Data Integrity Check for {name} ---")
    
    # Check column consistency (same features after SelectKBest)
    print("Column consistency:")
    if list(X_train.columns) == list(X_test.columns):
        print("✓ Selected columns are consistent between X_train and X_test")
    else:
        print("✗ Selected columns are not consistent")


    # Check for NaNs
    print("\nChecking for NaNs:")
    if X_train.isnull().sum().sum() > 0:
        print("✗ X_train has missing values")
    else:
        print("✓ X_train has no missing values")
        
    if X_test.isnull().sum().sum() > 0:
        print("✗ X_test has missing values")
    else:
        print("✓ X_test has no missing values")
        
    if y_train.isnull().sum() > 0:
        print("✗ y_train has missing values")
    else:
        print("✓ y_train has no missing values")
        
    if y_test.isnull().sum() > 0:
        print("✗ y_test has missing values")
    else:
        print("✓ y_test has no missing values")
    
    # Check data shape
    print("\nShape of data:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")



# Load the trained model
rf_model = joblib.load('random_forest_model.joblib')
# Load the scaler from the .joblib file
scaler = joblib.load('scaler.joblib')

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
    11: 'Wipro_bulb',}


# Function to shuffle and simulate real-time data feeding with row and probability printing
def simulate_real_time_feed_from_df(df, delay=1):
    # Shuffle the DataFrame to randomize row order
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for _, row in shuffled_df.iterrows():
        # Drop the 'Attack_type' or any target columns for prediction
        new_data_row = row.drop(['Attack_type', 'Attack'], errors='ignore')

        # Align with the feature columns used in training (X_test_selected_scaled_smote.columns)
        new_data_row = pd.DataFrame([new_data_row], columns= column_names_df.columns) #X_test_selected_scaled_smote

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Apply LabelEncoder to each categorical column
        for column in new_data_row.columns:
            if new_data_row[column].dtype == 'object' or new_data_row[column].dtype.name == 'category':
                new_data_row[column] = label_encoder.fit_transform(new_data_row[column]) 

        
        # Print the row's feature values before scaling and encoding
        print("\nFeature Values (Raw):")
        print(new_data_row)

        # Apply scaling to the row
        new_data_scaled = scaler.transform(new_data_row)
        new_data_scaled_df = pd.DataFrame(new_data_scaled, columns= column_names_df.columns) #X_test_selected_scaled_smote

        # Print the row's feature values after scaling
        #print("\nFeature Values (Scaled):")
        #print(new_data_scaled_df)

        # Get the prediction and probabilities
        prediction = rf_model.predict(new_data_scaled_df)
        prediction_proba = rf_model.predict_proba(new_data_scaled_df)

        # Convert numeric prediction to original class name using the manual label mapping
        prediction_label = label_mapping.get(prediction[0], "Unknown")

        # Print the prediction result and probabilities
        print(f"\nPredicted Attack Type: {prediction_label}")
        print(f"Prediction Probabilities: {prediction_proba[0]}")
        
        # Simulate real-time delay
        time.sleep(delay)

