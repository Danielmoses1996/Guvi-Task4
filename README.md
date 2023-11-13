# Guvi-Task4

# Data Understanding and Preprocessing

Load the dataset
Data Understanding
Identify variable types
Convert rubbish values in 'Material_Reference' to null
Treat reference columns as categorical variables
Data Preprocessing
Handle missing values
Treat outliers using IQR or Isolation Forest
For simplicity, let's use IQR
Identify and treat skewness using log transformation
Visualizing outliers and skewness

# Feature Engineering and Model Building

Feature Engineering
Example: Create a new feature by aggregating existing features
Drop highly correlated columns
Model Building and Evaluation
Split the dataset
Train and evaluate a random forest classifier
Model Evaluation

# Model GUI using Streamlit

Load the trained model
Sidebar for user input
Create input fields for each column
Add more input fields for other columns
Perform the same feature engineering and scaling steps
Making predictions
