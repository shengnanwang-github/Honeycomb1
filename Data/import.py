#!/usr/bin/env python
# conding: utf-8
# 1.Import libraries
# The first step in building the model is to import the necessary libraries.


import pandas as pd

# Load the dataset
# dataset_path = '/home/ubuntu/Honeycomb/Data/Data.csv'

dataset_path = '/home/ubuntu/Honeycomb/Data/Data.csv'
data = pd.read_csv(dataset_path)
df = pd.read_csv(dataset_path)
data_types = df.dtypes

# Get descriptive statistics for all columns
descriptive_statistics = df.describe(include='all')

# Get the number of missing values in each column
missing_values = df.isnull().sum()

# Get the number of unique values in each column
unique_values = df.nunique()

# Print the results
print("Data Types:")
print(data_types)
print("\nDescriptive Statistics:")
print(descriptive_statistics)
print("\nMissing Values:")
print(missing_values)
print("\nUnique Values:")
print(unique_values)
# Display the first few rows of the dataset to understand its structure

data.head()
