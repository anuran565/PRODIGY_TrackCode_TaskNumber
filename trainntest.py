import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
try:
    df=pd.read_csv("scaled_train.csv")
except FileNotFoundError:
    print("File Not Found !!! ")
    exit()

#Separate Features(X) and Target(Y)
X=df.drop('SalePrice', axis=1)  #Features all the columns except SalePrice
y=df['SalePrice']  #Target Variable


#Split the data
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#Print the Shapes to verify:
print("X_train shape : ",X_train.shape)
print("X_test shape : ",X_test.shape)
print("y_train shape : ",y_train.shape)
print("y_test shape : ",y_test.shape)

X_train['SalePrice']=y_train
X_test['SalePrice']=y_test

X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)


print("Training and testing data saved to csv files.")
