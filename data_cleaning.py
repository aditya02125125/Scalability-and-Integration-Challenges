import pandas as pd
import numpy as np

#load the dataset
df = pd.read_csv(r"cyber_ml_project/data/cicids2017_cleaned.csv")

print("Shape of the dataset:", df.shape)
print("\nColumns in the dataset:\n", df.columns)
print("\nFirst 5 rows of the dataset:\n", df.head())

#check for missing values
print("\nMissing values:\n", df.isnull().sum())

#remove duplicate rows
df.drop_duplicates()
print("\nAfter removing duplicates:", df.shape)

#handle infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("\nAfter removing NaN & Inf:", df.shape)

#check data types
print("\nData Types:\n", df.dtypes)

#Final cleaned dataset
df.to_csv("cyber_ml_project/data/cicids2017_cleaned_final.csv", index=False)
print("\nFinal Cleaned Dataset saved!")