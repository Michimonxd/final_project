
# IoT Attack Detection Final Project

## Overview
The IoT (Internet of Things) ecosystem is continuously growing, connecting billions of devices and creating new opportunities for innovative applications. However, this also introduces significant security challenges, as IoT devices are often vulnerable to various cyber-attacks. This project aims to develop a robust machine learning pipeline to detect and classify attack types in IoT network traffic using binary and multiclass classification techniques.

## Project Goals
- Perform exploratory data analysis (EDA) to understand IoT network behavior and identify key features.
- Clean and preprocess data to prepare it for machine learning modeling.
- Build, train, and evaluate machine learning models for detecting attack types.
- Develop a real-time application using Streamlit to predict attack types based on IoT data.

## Project Structure
This repository is organized as follows:

### Data Files
- **IOT.csv**: The initial raw dataset containing IoT device communication records, including normal and attack traffic.
- **IoT_cleaned.csv**: The cleaned version of the raw dataset, processed for better model performance.
- **test_data.csv**: A smaller dataset for testing and validating model predictions.

### Jupyter Notebooks
- **exploring_wrangling_binary.ipynb**: Notebook covering data exploration, preprocessing, and feature engineering for binary classification (attack vs. normal).
- **exploring_wrangling_multiclass.ipynb**: Notebook focusing on data preparation and feature engineering for multiclass classification (specific types of attacks).
- **ml_model_target_binary.ipynb**: Contains the code for building and evaluating machine learning models for binary classification.
- **ml_model_target_multiclass.ipynb**: Details the implementation and evaluation of models for multiclass classification.

### Python Scripts
- **functions.py**: A script containing reusable functions for data preprocessing, feature selection, and model evaluation.
- **functions/model_realtimedata.py**: Module to handle real-time data processing and communication between IoT devices and the model.

### Model Files
- **feature_names.joblib**: A serialized file containing the feature names used by the trained models.
- **random_forest_model_smote.joblib** & **rf_smote_model.pkl**: Pre-trained Random Forest models enhanced with Synthetic Minority Oversampling Technique (SMOTE) for handling imbalanced data.
- **scaler.joblib** & **scaler.pkl**: Pre-fitted scaler objects used to normalize input features during model inference.

### Other Files
- **streamlit_app.py**: The main script for deploying a Streamlit web app, enabling users to input IoT data and receive predictions about potential security threats.
- **conusion_matrix.jpg**: An image of a confusion matrix visualizing model performance, highlighting its accuracy in classifying attack types.

### Checkpoints and Backups
- **.ipynb_checkpoints/**: Folder containing checkpoint files for Jupyter notebooks.
- **Backup/final_project.ipynb**: A backup notebook to preserve the main project structure.

## Installation Guide
To set up the project locally, follow these steps:

### Prerequisites
Ensure that you have Python 3.x and the following libraries installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `streamlit`

## Usage Guide
### Exploratory Data Analysis
The `exploring_wrangling_binary.ipynb` and `exploring_wrangling_multiclass.ipynb` notebooks guide users through:
- Visualizing data distributions.
- Identifying and handling missing data.
- Feature correlation analysis to select the most influential features for modeling.

### Model Training and Evaluation
The `ml_model_target_binary.ipynb` and `ml_model_target_multiclass.ipynb` notebooks provide step-by-step instructions for:
- Splitting data into training and testing sets.
- Applying feature scaling and SMOTE for class balancing.
- Training models such as Random Forests and evaluating their performance with metrics like accuracy, F1-score, and confusion matrices.

### Real-Time Prediction App
The `streamlit_app.py` script deploys a web app that allows users to:
- Input data for real-time predictions.
- View predictions indicating whether input data suggests normal behavior or specific attack types.

## Model Details
- **Random Forest Classifier**: Selected for its robust performance in handling complex datasets and imbalanced classes. Enhanced with SMOTE to improve minority class representation.
- **Feature Engineering**: Key features were selected based on domain knowledge and correlation analysis.
- **Evaluation Metrics**:
  - **Confusion Matrix**: Visual representation of true positives, false positives, true negatives, and false negatives.
  - **Accuracy and F1-Score**: Metrics used to evaluate model performance, ensuring reliable predictions.

## Streamlit App Features
- **User Interface**: Intuitive design for easy input and result interpretation.
- **Model Integration**: Directly utilizes the pre-trained Random Forest model for live predictions.
- **Data Scaling**: Automatically applies the pre-trained scaler for consistent input transformation.

## Future Enhancements
- Integrate additional models (e.g., neural networks) for comparative analysis.
- Expand the dataset with more diverse IoT traffic data.
- Implement additional features in the Streamlit app for visualizing prediction trends over time.

## Contribution Guidelines
Contributions are welcome to improve the project or add new features:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Submit a pull request for review.


## Contact
For questions or support, please contact Michael Schilling(mailto:michael.schilling@hotmail.de).


