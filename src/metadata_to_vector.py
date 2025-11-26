import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 1. Load the Data
df = pd.read_csv('../dataset/metadata_raw.csv')
print("Original Data (only first 5 rows):")
print(df[['label', 'weight_gr', 'color', 'season', 'origin']].head())
print("-" * 30)

# --- NUMERICAL FEATURE (Weight) ---
# We must normalize weight so it is between [0, 1]
scaler = MinMaxScaler()
weight_feature = scaler.fit_transform(df[['weight_gr']]) 

# --- CATEGORICAL FEATURES (Color, Season, Origin) ---
# We define the specific categories to ensure fixed dimensionality, and handle unknown categories.
# Map unknown values to "Other" before encoding

# Define valid categories
valid_origins = ['Turkey', 'Spain', 'USA', 'Brazil', 'Ecuador', 'Other']
valid_colors = ['Red', 'Green', 'Yellow', 'Orange', 'Brown', 'Purple', 'Other']

# Replace unknown origin values with "Other"
df['origin'] = df['origin'].apply(lambda x: x if x in valid_origins else 'Other')
# Replace unknown color values with "Other"
df['color'] = df['color'].apply(lambda x: x if x in valid_colors else 'Other')

enc_color = OneHotEncoder(categories=[valid_colors], 
                          handle_unknown='ignore', sparse_output=False)

enc_season = OneHotEncoder(categories=[['Spring', 'Summer', 'Fall', 'Winter']], 
                           handle_unknown='ignore', sparse_output=False)

enc_origin = OneHotEncoder(categories=[valid_origins], 
                           handle_unknown='ignore', sparse_output=False)

# 2. Transform the raw columns
color_features = enc_color.fit_transform(df[['color']])
season_features = enc_season.fit_transform(df[['season']])
origin_features = enc_origin.fit_transform(df[['origin']])

# 3. Fusion the features
# Concatenate all features into a single vector per sample : [Weight (1)] + [Color (7)] + [Season (4)] + [Origin (6)]
categorical_numerical_vector = np.hstack([weight_feature, color_features, season_features, origin_features])

# 4. Print the results
print(f"Feature Vector Shape: {categorical_numerical_vector.shape}")
print(f"(Samples: {categorical_numerical_vector.shape[0]}, Dimensions: {categorical_numerical_vector.shape[1]})")

print("\nAll Categorical Features for All Samples:")
print(np.round(categorical_numerical_vector, 2)) # Rounding for readability