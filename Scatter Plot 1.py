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


# 2. Histograms for Numerical Features
year_columns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
numerical_cols = df.select_dtypes(include=np.number).columns
numerical_cols = numerical_cols.drop(year_columns, errors='ignore')

df[numerical_cols].hist(figsize=(20,15))
plt.suptitle("Histograms of Numerical Features (Excluding Year Columns)", fontsize=16)
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))  # 5 rows, 4 columns of subplots

# Flatten the axes array for easier iteration
axes = axes.flatten()

for i, feature in enumerate(numerical_cols):
    if i < len(axes): # Check if there are enough subplots
        ax = axes[i]
        df[feature].hist(ax=ax)
        ax.set_title(feature)

plt.tight_layout()
plt.show()


# 3. Scatter Plots (Numerical Features vs. SalePrice)
for feature in numerical_cols:
    plt.figure(figsize=(8,6))
    plt.scatter(df[feature], df['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(f'Scatter Plot of {feature} vs. SalePrice')
    plt.grid(True)
    snb.regplot(x=df[feature], y=df['SalePrice'], scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    plt.show()

# 4. Correlation Matrix Heatmap
correlation_matrix = df[numerical_cols.append(pd.Index(['SalePrice']))].corr() #Including SalePrice in the correlation matrix
plt.figure(figsize=(15, 12))
snb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap (Numerical Features and SalePrice)')
plt.show()
