# This script is a combination of metadata_to_vector.py and description_to_vector.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer
import os

# --- CONFIGURATION ---
METADATA_PATH = '../dataset/raw/metadata.csv'
DESCRIPTION_PATH = '../dataset/raw/description.csv'
OUTPUT_FOLDER = '../dataset/processed'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("--- 1. LOADING AND ALIGNING DATA ---")
# Load both CSVs
df_meta = pd.read_csv(METADATA_PATH)
df_desc = pd.read_csv(DESCRIPTION_PATH)

# Merge on 'ID' to ensure rows align perfectly
df_merged = pd.merge(df_meta, df_desc, on='ID')

print(f"Merged Data Shape: {df_merged.shape}")
print(f"Sample IDs: {df_merged['ID'].head().tolist()}")
print("-" * 30)


print("--- 2. EXTRACTING NUMERICAL & CATEGORICAL FEATURES ---")

# A. Numerical (Weight)
scaler = MinMaxScaler()
weight_feature = scaler.fit_transform(df_merged[['weight_gr']]) 

# B. Categorical (Color, Season, Origin)
valid_origins = ['Turkey', 'Spain', 'USA', 'Brazil', 'Ecuador', 'Other']
valid_colors = ['Red', 'Green', 'Yellow', 'Orange', 'Brown', 'Purple', 'Other']

# Handling unknown values
df_merged['origin'] = df_merged['origin'].apply(lambda x: x if x in valid_origins else 'Other')
df_merged['color'] = df_merged['color'].apply(lambda x: x if x in valid_colors else 'Other')

# Encoders
enc_color = OneHotEncoder(categories=[valid_colors], handle_unknown='ignore', sparse_output=False)
enc_season = OneHotEncoder(categories=[['Spring', 'Summer', 'Fall', 'Winter']], handle_unknown='ignore', sparse_output=False)
enc_origin = OneHotEncoder(categories=[valid_origins], handle_unknown='ignore', sparse_output=False)

# Transform
color_features = enc_color.fit_transform(df_merged[['color']])
season_features = enc_season.fit_transform(df_merged[['season']])
origin_features = enc_origin.fit_transform(df_merged[['origin']])

# C. Concatenate Metadata Features
# Dims: 1 (Weight) + 7 (Color) + 4 (Season) + 6 (Origin) = 18 Dimensions
meta_vector = np.hstack([weight_feature, color_features, season_features, origin_features])
print(f"Metadata Vector Shape: {meta_vector.shape}")


print("\n--- 3. EXTRACTING TEXT FEATURES ---")
# (Logic adapted from description_to_vector.py)

print(f"Loading Sentence Transformer: {EMBEDDING_MODEL}...")
text_model = SentenceTransformer(EMBEDDING_MODEL)

descriptions = df_merged['description'].tolist()
# Dims: 384 Dimensions
text_vector = text_model.encode(descriptions, convert_to_numpy=True)
print(f"Text Vector Shape: {text_vector.shape}")


print("\n--- 4. FUSION AND SAVING ---")
# Fuse the two modalities 
# Total Dims: 18 + 384 = 402
X_final = np.hstack([meta_vector, text_vector])

# Extract Ground Truth Labels
y_final = df_merged['label'].values

print(f"Final X (Features) Shape: {X_final.shape}")
print(f"Final y (Labels) Shape:   {y_final.shape}")

# Dimensionality Check [cite: 14]
if 10 < X_final.shape[1] < 500:
    print(f"✅ Dimensionality check passed: {X_final.shape[1]} dimensions.")
else:
    print(f"⚠️ Warning: Dimensionality is {X_final.shape[1]}. Check requirements.")

# Save to files
x_path = os.path.join(OUTPUT_FOLDER, 'X_features.npy')
y_path = os.path.join(OUTPUT_FOLDER, 'y_labels.npy')

np.save(x_path, X_final)
np.save(y_path, y_final)

print(f"\nSuccessfully saved fused vectors to:\n -> {x_path}\n -> {y_path}")