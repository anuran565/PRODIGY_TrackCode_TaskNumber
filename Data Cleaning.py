import pandas as pd
import numpy as np

# 1. Load the CSV file (Corrected: df is defined within the try block)
try:
    df = pd.read_csv("train.csv")  # Load your CSV
except FileNotFoundError:
    print("File not found. Check the path.")
    exit()  # It's best to exit if the file isn't found, otherwise, the rest of the code will likely fail

print("Initial DataFrame Info:")
print(df.info())

LotFrontage = 'LotFrontage'  # It's good practice to store column names in variables

# 2. Check for whitespace in column names (Improved)
df.columns = df.columns.str.strip()  # Strip all column names at once (more efficient)
LotFrontage = LotFrontage.strip()  # Strip the variable too

# 3. Check if the column exists (Improved)
if LotFrontage not in df.columns:
    print(f"Error: Column '{LotFrontage}' not found.")
    exit()

# 4. Check data type and convert if needed (Improved)
print(f"Data type of '{LotFrontage}': {df[LotFrontage].dtype}") #Corrected the column name

if not pd.api.types.is_numeric_dtype(df[LotFrontage]): #Corrected the column name
    try:
        df[LotFrontage] = pd.to_numeric(df[LotFrontage], errors='coerce')  # Use errors='coerce' to handle non-numeric values gracefully
        print(f"Successfully converted '{LotFrontage}' to numeric.")
    except ValueError:
        print(f"Error: Cannot convert '{LotFrontage}' to numeric. Non-numeric values found.")
        non_numeric_values = df[pd.to_numeric(df[LotFrontage], errors='coerce').isnull()]  # Identify non-numeric values
        print("Non-numeric values:", non_numeric_values)
        # Handle the non-numeric values (replace, remove, etc.)
        # For now, let's just exit to avoid further errors
        exit()

# 5. Impute with median
median_value = df[LotFrontage].median() #Corrected the column name
df[LotFrontage].fillna(median_value, inplace=True) #Corrected the column name
print("Data Frame after imputation : ")
print(df.info())

# 6. Handling False/Incorrect Values (If Any)

# This part depends entirely on what you consider "false" or "incorrect" in your data.
# You'll need to define the criteria based on your data and domain knowledge.

# Example 1: Removing rows where 'SalePrice' is negative (if that's an error)
df = df[df['SalePrice'] >= 0]  # Keep rows where 'SalePrice' is non-negative

# Example 2: Replacing unrealistic values in 'LotArea' (example)
df['LotArea'] = np.where(df['LotArea'] > 100000, np.nan, df['LotArea']) #Replacing if the LotArea is > 100000 with NaN
median_lotarea = df['LotArea'].median() #Calculating the median of LotArea
df['LotArea'].fillna(median_lotarea, inplace=True) #Filling NaN values with median of LotArea

# Example 3: Replacing values in 'GarageYrBlt' if they are before 1900
df['GarageYrBlt'] = np.where(df['GarageYrBlt'] < 1900, np.nan, df['GarageYrBlt']) #Replacing if the GarageYrBlt is < 1900 with NaN
median_garage = df['GarageYrBlt'].median() #Calculating the median of GarageYrBlt
df['GarageYrBlt'].fillna(median_garage, inplace=True) #Filling NaN values with median of GarageYrBlt

print("\nDataFrame Info after handling incorrect values (examples):")
print(df.info())

# 7. Save the Cleaned Dataset

df.to_csv("cleaned_train.csv", index=False)  # Save to a new CSV file
print("\nCleaned dataset saved to 'cleaned_train.csv'")




