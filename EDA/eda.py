#------------------------------------------------------------the distribution of mental health issues across different countries-----------------------------------------------------------
#!/usr/bin/env python
# coding: utf-8
# We will create visualizations to illustrate the distribution of mental health issues across different countries.

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
data = pd.read_csv('/home/ubuntu/Honeycomb/Data/clean_data.csv')

# Setting the aesthetics for the plots
sns.set(style="whitegrid")

# Geographical Analysis: Distribution of mental health issues across countries
country_counts = data['Country'].value_counts()
significant_countries = country_counts[country_counts > 10]  # Countries with more than 10 respondents
significant_data = data[data['Country'].isin(significant_countries.index)]

# Plotting the distribution of mental health issues in these countries
plt.figure(figsize=(15, 6))
sns.countplot(x='Country', data=significant_data, palette="Set2", order=significant_countries.index)
plt.xticks(rotation=45)
plt.title('Distribution of Mental Health Issues Across Countries')
plt.xlabel('Country')
plt.ylabel('Number of Respondents')

# Create a directory for saving plots if it doesn't exist
save_dir = "/home/ubuntu/Honeycomb/EDA"
os.makedirs(save_dir, exist_ok=True)

# Save the plot
plt.savefig(os.path.join(save_dir, "Distribution.png"))

# Show the plot
plt.show()

#------------------------------------------------------------------age group analysis-------------------------------------------------------------

# Creating age groups
age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]
data['Age_Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

# Plotting the distribution of age groups
plt.figure(figsize=(12, 6))
sns.countplot(x='Age_Group', data=data, palette="Set1")
plt.title('Distribution of Respondents Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of Respondents')

# Create a directory for saving plots if it doesn't exist
save_dir = "/home/ubuntu/Honeycomb/EDA"
os.makedirs(save_dir, exist_ok=True)

# Save the plot
plt.savefig(os.path.join(save_dir, "Age_Distribution.png"))

# Show the plot
plt.show()

#------------------------------------------------------------------------Correlation between Family History and Treatment-------------------------------------------------------

# Creating a crosstab for family history and treatment
family_history_treatment = pd.crosstab(data['family_history'], data['treatment'])

# Plotting the correlation between family history and treatment
plt.figure(figsize=(8, 5))
sns.heatmap(family_history_treatment, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Correlation between Family History and Treatment')
plt.xlabel('Treatment')
plt.ylabel('Family History of Mental Health Issues')

# Create a directory for saving plots if it doesn't exist
save_dir = "/home/ubuntu/Honeycomb/EDA"
os.makedirs(save_dir, exist_ok=True)

# Save the plot
plt.savefig(os.path.join(save_dir, "Family_History_Correlation.png"))

# Show the plot
plt.show()

#----------------------------------------------------------------------------------heatmap of variables correlation with treatment-----------------------------------------------------

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '/home/ubuntu/Honeycomb/Data/clean_data.csv'
data = pd.read_csv(file_path)

# Convert categorical features to numerical
categorical_features = ['Gender', 'Country', 'family_history', 'treatment', 'work_interfere',
                        'benefits', 'care_options', 'wellness_program', 'seek_help',
                        'phys_health_interview', 'mental_health_interview', 'mental_vs_physical']

# Initialize a dictionary to store LabelEncoders for each categorical feature
encoders = {feature: LabelEncoder() for feature in categorical_features}

# Use the LabelEncoder instance from the dictionary to fit and transform the data
for feature in categorical_features:
    data[feature] = encoders[feature].fit_transform(data[feature].astype(str))

# Define features and target variable
features = ['Age', 'Gender', 'Country', 'family_history', 'work_interfere',
            'benefits', 'care_options', 'wellness_program', 'seek_help',
            'phys_health_interview', 'mental_health_interview', 'mental_vs_physical']

# Calculate the correlation matrix
corr = data[features + ['treatment']].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title('Heatmap of Variables Correlation with Treatment')

# Create a directory for saving plots if it doesn't exist
save_dir = "/home/ubuntu/Honeycomb/EDA"
os.makedirs(save_dir, exist_ok=True)

# Save the plot
plt.savefig(os.path.join(save_dir, "Variables_Correlation_Heatmap.png"))

# Show the plot
plt.show()

