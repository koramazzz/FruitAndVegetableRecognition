# This script combines metadata_to_vector.py, description_to_vector.py, and image_to_vector.py

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# --- CONFIGURATION ---
METADATA_PATH = '../dataset/raw/metadata.csv'
DESCRIPTION_PATH = '../dataset/raw/description.csv'
IMAGES_BASE_PATH = '../dataset/images/original'
OUTPUT_FOLDER = '../dataset/processed'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Image feature extraction parameters
COLOR_HIST_BINS = 64  # Bins per channel for color histogram
LBP_RADIUS = 3
LBP_N_POINTS = 24

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


print("\n--- 4. EXTRACTING IMAGE FEATURES ---")
# (Logic adapted from image_to_vector.py)

def extract_color_histogram(image, bins=COLOR_HIST_BINS):
    """Extract color histogram from RGB image."""
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    # Normalize histograms
    hist_r = hist_r / (hist_r.sum() + 1e-7)
    hist_g = hist_g / (hist_g.sum() + 1e-7)
    hist_b = hist_b / (hist_b.sum() + 1e-7)
    
    color_features = np.hstack([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
    return color_features


def extract_lbp_features(image_gray):
    """Extract Local Binary Pattern (LBP) features."""
    lbp = local_binary_pattern(image_gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    return hist


def extract_glcm_features(image_gray):
    """Extract Gray-Level Co-occurrence Matrix (GLCM) features."""
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                       levels=256, symmetric=True, normed=True)
    
    properties = ['contrast', 'energy', 'homogeneity', 'correlation']
    all_features = []
    
    # Average features across all angles (4 dimensions)
    for prop in properties:
        prop_values = graycoprops(glcm, prop)
        all_features.append(np.mean(prop_values))
    
    # Individual features for first 3 angles and first 3 properties (9 dimensions)
    for angle_idx in range(3):
        for prop_idx in range(3):
            prop_values = graycoprops(glcm, properties[prop_idx])
            all_features.append(prop_values[0, angle_idx])
    
    return np.array(all_features)


def extract_statistical_features(image):
    """Extract statistical features from RGB image."""
    features = []
    
    for channel_idx in range(3):  # R, G, B channels
        channel = image[:, :, channel_idx]
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.min(channel),
            np.max(channel),
            np.median(channel),
            np.var(channel),
            np.percentile(channel, 75) - np.percentile(channel, 25)  # IQR
        ])
    
    return np.array(features)


def extract_image_features(image_path):
    """Extract all features from a single image."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for texture features
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract features
    color_hist = extract_color_histogram(image_rgb)
    lbp_features = extract_lbp_features(image_gray)
    glcm_features = extract_glcm_features(image_gray)
    statistical_features = extract_statistical_features(image_rgb)
    
    # Combine all features
    all_features = np.hstack([
        color_hist,
        lbp_features,
        glcm_features,
        statistical_features
    ])
    
    return all_features


# Extract image features for all samples
print(f"Processing {len(df_merged)} images...")
image_features_list = []
failed_ids = []

for idx, row in df_merged.iterrows():
    sample_id = row['ID']
    label = row['label']
    
    # Construct image path: dataset/images/original/{label}/{ID}.jpg
    image_path = os.path.join(IMAGES_BASE_PATH, label.lower(), f"{sample_id}.jpg")
    
    try:
        features = extract_image_features(image_path)
        image_features_list.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df_merged)} images...")
            
    except Exception as e:
        print(f"  ⚠️ Error processing {sample_id}: {e}")
        failed_ids.append(sample_id)
        # Use zero vector as placeholder
        if len(image_features_list) > 0:
            image_features_list.append(np.zeros_like(image_features_list[0]))
        else:
            # If first image fails, use expected dimension
            image_features_list.append(np.zeros(COLOR_HIST_BINS * 3 + 59 + 13 + 21))

# Convert to numpy array
image_vector = np.array(image_features_list)
print(f"Image Vector Shape: {image_vector.shape}")
print(f"Successfully processed {len(image_features_list) - len(failed_ids)}/{len(df_merged)} images")
if failed_ids:
    print(f"Failed IDs: {failed_ids}")


print("\n--- 5. FUSION AND SAVING ---")
# Fuse all three modalities 
# Total Dims: 18 (Metadata) + 384 (Text) + 285 (Image) = 687
X_final = np.hstack([meta_vector, text_vector, image_vector])

# Extract Ground Truth Labels
y_final = df_merged['label'].values

print(f"Final X (Features) Shape: {X_final.shape}")
print(f"Final y (Labels) Shape:   {y_final.shape}")

# Dimensionality Check [cite: 14]
# Note: With image features, total is 687 dimensions (exceeds 500 limit)
# Consider using PCA or feature selection if needed
if 10 < X_final.shape[1] < 500:
    print(f"Dimensionality check passed: {X_final.shape[1]} dimensions.")
else:
    print(f"Warning: Dimensionality is {X_final.shape[1]}. Check requirements.")

# Save to files
x_path = os.path.join(OUTPUT_FOLDER, 'X_features.npy')
y_path = os.path.join(OUTPUT_FOLDER, 'y_labels.npy')

np.save(x_path, X_final)
np.save(y_path, y_final)

print(f"\nSuccessfully saved fused vectors to:\n -> {x_path}\n -> {y_path}")