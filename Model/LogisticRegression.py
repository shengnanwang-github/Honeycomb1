import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Load the dataset
file_path = '/home/ubuntu/Honeycomb/Data/clean_data.csv'
data = pd.read_csv(file_path, parse_dates=['Timestamp'])

# Select features and the target variable
features = ['Age', 'Gender', 'Country', 'self_employed', 'family_history', 'remote_work', 'tech_company']
target = 'treatment'

# Data preprocessing
X = data[features]
y = data[target].apply(lambda x: 1 if x == 'Yes' else 0)

# Perform one-hot encoding on categorical features
categorical_features = ['Gender', 'Country', 'self_employed', 'family_history', 'remote_work', 'tech_company']
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = onehot_encoder.fit_transform(X[categorical_features])

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# # # Build a Random Forest model for feature importance analysis
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)



# Logistic Regression Analysis
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Evaluate the Logistic Regression model
y_pred_log = log_model.predict(X_test)
print("Logistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_log))
