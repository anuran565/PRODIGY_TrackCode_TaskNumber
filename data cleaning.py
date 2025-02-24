import pandas as pd


#Load the dataset
try:
    df=pd.read_csv("Mall_Customers.csv")
    print("The Original Dataset : ")
    
except FileNotFoundError:
    print("File Not Found.")


print(df.info())  #Display the first few rows
print(df.head())  #Get information about data types and missing values
print(df.describe()) #Get the descriptive statistics of numerical columns
print(df.shape)  #Get the number of rows and columns

print(df.isnull().sum()) #Number of missing values per column
print(df.isnull().any()) #True if there are any missing values,otherwise False
print(df.isnull().sum().sum()) #Total number of missing values in the DataFrame

