import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    df=pd.read_csv("cleaned_train.csv")
except FileNotFoundError:
    print("Not able to load the cleaned data set.")
    exit()

print("The cleaned data set : ")
print(df.info())

X=df.drop('SalePrice',axis=1) # Features all columns except Sale Price
y=df['SalePrice'] #Target variable(SalePrice)

numerical_cols=X.select_dtypes(include=np.number).columns

year_columns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
numerical_cols = numerical_cols.drop(year_columns, errors='ignore')

scaler=StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols, index=X.index)

X[numerical_cols] = X_scaled_df

print("\nScaled Features (First 5 rows):")
print(X.head())
print(X.tail())

X['SalePrice'] = y  # Add the target variable back to the data
X.to_csv("scaled_train.csv", index=False)
print("\nScaled data saved to 'scaled_train.csv'")

print("\nMean and Standard Deviation of Scaled Columns:")
print(X[numerical_cols].mean())  
print(X[numerical_cols].std()) 
