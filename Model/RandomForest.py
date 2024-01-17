# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np

# Load the dataset
file_path = '/home/ubuntu/Honeycomb/Data/clean_data.csv'
data = pd.read_csv(file_path, parse_dates=['Timestamp'])

# Convert categorical features to numerical
categorical_features = ['family_history', 'treatment', 'work_interfere',
                        'benefits', 'care_options', 'wellness_program', 'seek_help']

# Initialize a dictionary to store LabelEncoders for each categorical feature
encoders = {feature: LabelEncoder() for feature in categorical_features}

for feature in categorical_features:
    # Use the LabelEncoder instance from the dictionary to fit and transform the data
    data[feature] = encoders[feature].fit_transform(data[feature].astype(str))

# Define features and target variable
X = data[['family_history', 'work_interfere', 'benefits',
          'care_options', 'wellness_program', 'seek_help']]
y = data['treatment']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{report}')
