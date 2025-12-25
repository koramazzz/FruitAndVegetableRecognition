"""Quick script to find what sample corresponds to a training set index."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "processed"
RAW_DIR = BASE_DIR / "dataset" / "raw"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Load data
X = np.load(DATA_DIR / "X_final.npy")
y_raw = np.load(DATA_DIR / "y_final.npy", allow_pickle=True)

# Load metadata
df_meta = pd.read_csv(RAW_DIR / "metadata.csv")
df_desc = pd.read_csv(RAW_DIR / "description.csv")
df = pd.merge(df_meta, df_desc, on='ID')

# Same split as svm_scratch.py
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
train_indices, test_indices = train_test_split(
    np.arange(len(X)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Get the index to search
import sys
search_idx = int(sys.argv[1])

if search_idx >= len(train_indices):
    print(f"Error: Index {search_idx} is out of range (training set size: {len(train_indices)})")
    sys.exit(1)

original_idx = train_indices[search_idx]
sample = df.iloc[original_idx]

print(f"    Training set index : {search_idx}")
print(f"Original dataset index : {original_idx}")
print(f"                    ID : {sample['ID']}")

