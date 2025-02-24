import pandas as pd

try:
    df = pd.read_csv("cleaned_train.csv")  
except FileNotFoundError:
    print("File not found. Check the path.")
    exit()

# Example 1: Total Bathrooms (weighted)
df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

# Example 2: Bathrooms per Bedroom Ratio
df['BathroomsPerBedroom'] = df['TotalBathrooms'] / (df['BedroomAbvGr'] + 1e-6)  # Adding a small value to avoid division by zero

# Example 3: Years Since Built (assuming 'YrSold' is the year of sale)
current_year = 2024 #Or you can extract it from YrSold column
df['YearsSinceBuilt'] = current_year - df['YearBuilt']

# Example 4: Total Square Footage
df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']

df['GarageAreaToLotArea'] = df['GarageArea'] / (df['LotArea'] + 1e-6)
df['FirstFloorSFToTotalSF'] = df['1stFlrSF'] / (df['TotalSF'] + 1e-6)
df['PoolAreaToLotArea'] = df['PoolArea'] / (df['LotArea'] + 1e-6)
df['FireplacesToTotalRooms'] = df['Fireplaces'] / (df['TotRmsAbvGrd'] + 1e-6)


df['SalePrice']
df.to_csv("feature_engineering_train.csv", index=False)


print(df[['TotalBathrooms', 'BathroomsPerBedroom', 'YearsSinceBuilt', 'TotalSF', 'SalePrice','GarageAreaToLotArea',
          'FirstFloorSFToTotalSF','PoolAreaToLotArea','FireplacesToTotalRooms']].describe())
