import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Load the dataset
try:
    df=pd.read_csv("Mall_Customers.csv")
    print("The Original Dataset : ")
    print(df.info())
    
except FileNotFoundError:
    print("File Not Found.")
    exit()


# Separate numerical and categorical columns correctly
numerical_columns = df.select_dtypes(include=np.number).columns
categorical_columns = df.select_dtypes(exclude=np.number).columns  # Get categorical columns


df[numerical_columns].hist(figsize=(10,10))
plt.show()


# Scatter plots to show the relationship between numerical features
# Create a scatter plot for each pair of numerical columns
num_cols = len(numerical_columns)
for i in range(num_cols):
    for j in range(i + 1, num_cols):  # Avoid plotting the same pair twice
        plt.figure(figsize=(8, 6))
        plt.scatter(df[numerical_columns[i]], df[numerical_columns[j]])
        plt.xlabel(numerical_columns[i])
        plt.ylabel(numerical_columns[j])
        plt.title(f"Scatter plot: {numerical_columns[i]} vs {numerical_columns[j]}")
        plt.show()

#Correlation heatmap to identify the relationships between the features
correlation = df[numerical_columns].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.show()
