import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATASET
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "data", "cicids2017_cleaned.csv")
df = pd.read_csv(file_path)
print("Dataset Loaded:", df.shape)

# =========================
# FIX COLUMN NAMES
# =========================
df.columns = df.columns.str.strip()

print("\nALL COLUMNS:\n", df.columns.tolist())

# =========================
# DETECT LABEL COLUMN
# =========================
possible_labels = ['Label', 'label', 'attack', 'Attack', 'class', 'Class', 'Category']

label_col = None
for col in df.columns:
    if col in possible_labels:
        label_col = col
        break

# If not found → assume last column
if label_col is None:
    label_col = df.columns[-1]
    print(f"\nNo standard label found, using last column: {label_col}")

# Rename to standard
df.rename(columns={label_col: 'Label'}, inplace=True)

print("\nFinal Label Column:", label_col)
print("\nUnique Labels:\n", df['Label'].unique())

# =========================
# CONVERT LABEL TO BINARY
# =========================
df['Label'] = df['Label'].apply(
    lambda x: 0 if str(x).strip().lower() == 'normal traffic' else 1)
print("\nLabel Distribution:\n", df['Label'].value_counts())

# =========================
# BALANCE DATASET (SAFE VERSION)
# =========================

df_0 = df[df['Label'] == 0]
df_1 = df[df['Label'] == 1]

print("\nBefore Balancing:")
print(df['Label'].value_counts())

# check if both classes exist
if len(df_0) == 0 or len(df_1) == 0:
    print("\nOnly one class present. Skipping balancing.")
else:
    min_count = min(len(df_0), len(df_1))

    df_0 = df_0.sample(min_count, random_state=42)
    df_1 = df_1.sample(min_count, random_state=42)

    df = pd.concat([df_0, df_1])

    print("\nAfter Balancing:")
    print(df['Label'].value_counts())

# =========================
# REMOVE NON-NUMERIC COLUMNS
# =========================
non_numeric_cols = df.select_dtypes(exclude=['number']).columns
print("\nNon-numeric columns:\n", non_numeric_cols)

df = df.drop(columns=non_numeric_cols)

print("\nAfter removing non-numeric columns:", df.shape)

# =========================
# SPLIT FEATURES & TARGET
# =========================
X = df.drop('Label', axis=1)
y = df['Label']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling completed!")

# =========================
# SAVE PROCESSED DATA
# =========================

# get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# correct data folder path
data_dir = os.path.join(current_dir, "..", "data")

# save files
np.save(os.path.join(data_dir, "X_scaled.npy"), X_scaled)
np.save(os.path.join(data_dir, "y.npy"), y)

print("\nProcessed data saved successfully!")