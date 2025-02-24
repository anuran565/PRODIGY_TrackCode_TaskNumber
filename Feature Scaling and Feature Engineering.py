import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
try:
    df=pd.read_csv("Mall_Customers.csv")
    print("Data Loaded Successfully")
except FileNotFoundError:
    print("File Not Found Error")
    exit()

#Check whether any missing values are there or not
print(df.isnull().sum())
#Ensures Gender Column is categorical 
print(df.dtypes)

#Feature Engineering
#Gender Encoding
#Gender column is categorical.We must convert it into a numerical form before using it-in K-means.
df=pd.get_dummies(df, columns=['Gender'], drop_first=True)#drop first avoids multicollinearity and this creates a Gender_Male Column(1 for male,0 for female)




#Interaction Feature
df['Income_Spending'] = df['Annual Income (k$)'] * df['Spending Score (1-100)']

#Customer ID is just an identifier and should not be used in K-means clustering algorithm
df=df.drop('CustomerID', axis=1)

#Feature Scaling
#Standardization(Z-Score Normalization)
scaler = StandardScaler()
features_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Income_Spending']  # Include all relevant features
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

#Visualizing Distribution Features after scaling
for col in features_to_scale:
    plt.figure()
    sns.histplot(df[col], kde=True) #Use histplot for distributions

    
    plt.title(f'Distributions of {col} (Scaled)')
    plt.show()

#To verify that the transformations are applied correctly or not
print(df.head())

df.to_csv("Scaled_Mall_Customers.csv", index=False)
print("Scaled Data Saved Successfully.")
