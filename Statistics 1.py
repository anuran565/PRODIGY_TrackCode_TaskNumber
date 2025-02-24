import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb

#Loading the dataset
try:
    df=pd.read_csv("cleaned_train.csv")
except FileNotFoundError:
    print("Not able to load the cleaned data set.")
    exit()

print("The cleaned data set : ")
print(df.info())


#Identify year columns (adjust the list if you have other year-related columns)
year_columns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']  # Add any other year-related columns

# Select numerical columns *excluding* the year columns
numerical_cols = df.select_dtypes(include=np.number).columns  # Select numerical columns
numerical_cols = numerical_cols.drop(year_columns, errors='ignore') # Drop the year columns from the list of numerical columns

print("Numerical columns (excluding year columns):")
print(numerical_cols)

#Calculate the descriptive statistics
descriptive_stats=df[numerical_cols].describe(include='all')
descriptive_stats.loc['variance']=df[descriptive_stats].var()


#Calculate the specific statistics
mean_values=df[numerical_cols].mean()
median_values=df[numerical_cols].median()
std_values=df[numerical_cols].std()
variance_values=df[numerical_cols].var()


#Create a new dataframe to store the new results
custom_stats=pd.DataFrame({
    'mean':mean_values,
    'median':median_values,
    'standard deviation':std_values,
    'variance':variance_values
    })

print("Descriptive Statistics : ")
print(descriptive_stats)

print("Specific Statistics : ")
print(custom_stats)

