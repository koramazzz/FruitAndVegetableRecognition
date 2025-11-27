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
IMAGES_BASE_PATH = '../dataset/images/original'  # Ensure this points to the folder containing 'Banana', 'Tomato' etc.
OUTPUT_FOLDER = '../dataset/processed'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Image Feature Settings
COLOR_HIST_BINS = 64
LBP_RADIUS = 3
LBP_N_POINTS = 24

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS: IMAGE EXTRACTION
# ==========================================

def extract_color_histogram(image, bins=COLOR_HIST_BINS):
    # Calculate histogram for each channel
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    # Normalize to handle different image sizes
    hist_r = hist_r / (hist_r.sum() + 1e-7)
    hist_g = hist_g / (hist_g.sum() + 1e-7)
    hist_b = hist_b / (hist_b.sum() + 1e-7)
    
    return np.hstack([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])

def extract_lbp_features(image_gray):
    """
    Extract Local Binary Pattern (LBP) features.
    Returns: Uniform LBP histogram (59 dimensions)
    """
    lbp = local_binary_pattern(image_gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    # Uniform patterns have 59 bins: 0-58
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    return hist

def extract_glcm_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                       levels=256, symmetric=True, normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation']
    all_features = []
    
    for prop in properties:
        all_features.append(np.mean(graycoprops(glcm, prop))) # Average
    
    # Add specific angle details
    for angle_idx in range(3):
        for prop_idx in range(3):
            all_features.append(graycoprops(glcm, properties[prop_idx])[0, angle_idx])
            
    return np.array(all_features)

def extract_statistical_features(image):
    """
    Extract statistical features from RGB image.
    Returns: 21 features (7 stats * 3 channels)
    """
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

def get_image_vector(image_path):
    # Load and validate
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Combine all image features
    return np.hstack([
        extract_color_histogram(image_rgb),
        extract_lbp_features(image_gray),
        extract_glcm_features(image_gray),
        extract_statistical_features(image_rgb)
    ])

# ==========================================
# MAIN PIPELINE
# ==========================================

print("--- 1. LOADING AND ALIGNING DATA ---")
# Load CSVs
df_meta = pd.read_csv(METADATA_PATH)
df_desc = pd.read_csv(DESCRIPTION_PATH)

# CRITICAL: Merge on 'ID' to align Descriptions with Metadata/Images
df = pd.merge(df_meta, df_desc, on='ID')
print(f"Total Aligned Samples: {len(df)}")
print(f"Sample IDs (First 3): {df['ID'].head(3).tolist()}")
print("-" * 30)


print("\n--- 2. PROCESSING METADATA (NUMERICAL/CATEGORICAL) ---")
# A. Numerical (Weight)
scaler_weight = MinMaxScaler()
weight_feature = scaler_weight.fit_transform(df[['weight_gr']]) 

# B. Categorical (Color, Season, Origin)
valid_origins = ['Turkey', 'Spain', 'USA', 'Brazil', 'Ecuador', 'Other']
valid_colors = ['Red', 'Green', 'Yellow', 'Orange', 'Brown', 'Purple', 'Other']

# Handle unknown values
df['origin'] = df['origin'].apply(lambda x: x if x in valid_origins else 'Other')
df['color'] = df['color'].apply(lambda x: x if x in valid_colors else 'Other')

# Encode
enc_color = OneHotEncoder(categories=[valid_colors], handle_unknown='ignore', sparse_output=False)
enc_season = OneHotEncoder(categories=[['Spring', 'Summer', 'Fall', 'Winter']], handle_unknown='ignore', sparse_output=False)
enc_origin = OneHotEncoder(categories=[valid_origins], handle_unknown='ignore', sparse_output=False)

color_features = enc_color.fit_transform(df[['color']])
season_features = enc_season.fit_transform(df[['season']])
origin_features = enc_origin.fit_transform(df[['origin']])

# Combine Metadata
meta_vector = np.hstack([weight_feature, color_features, season_features, origin_features])
print(f"Metadata Features Shape: {meta_vector.shape}")


print("\n--- 3. PROCESSING TEXT DESCRIPTIONS ---")
print(f"Loading Model: {EMBEDDING_MODEL}...")
text_model = SentenceTransformer(EMBEDDING_MODEL)
descriptions = df['description'].tolist()
text_vector_raw = text_model.encode(descriptions, convert_to_numpy=True)

scaler_text = MinMaxScaler()
text_vector = scaler_text.fit_transform(text_vector_raw)
print(f"Text Features Shape: {text_vector.shape} (normalized)")


print("\n--- 4. PROCESSING IMAGES ---")
image_features_list = []
failed_indices = []
feature_dim = None

for idx, row in df.iterrows():
    sample_id = row['ID']
    label = row['label']
    
    # Construct path: dataset/images/original/banana/banana_001.jpg
    image_path = os.path.join(IMAGES_BASE_PATH, label.lower(), f"{sample_id}.jpg")
    
    try:
        feats = get_image_vector(image_path)
        if feature_dim is None: feature_dim = len(feats)
        image_features_list.append(feats)
    except Exception as e:
        print(f"  Error loading {sample_id}: {e}")
        failed_indices.append(idx)
        # Placeholder: Zero vector
        if feature_dim: image_features_list.append(np.zeros(feature_dim))
        else: image_features_list.append(None) # Handle if very first fails

# Convert list to array
image_vector_raw = np.array(image_features_list)

# Normalize Image Features (Critical step!)
scaler_img = MinMaxScaler()
image_vector = scaler_img.fit_transform(image_vector_raw)
print(f"Image Features Shape: {image_vector.shape}")


print("\n--- 5. FINAL FUSION AND SAVING ---")
# Fuse: [Metadata (18)] + [Text (384)] + [Images (189)]
X_final = np.hstack([meta_vector, text_vector, image_vector])
y_final = df['label'].values

# Stats
print(f"\nFINAL STATISTICS:")
print(f"Total Samples: {X_final.shape[0]}")
print(f"Total Dimensions: {X_final.shape[1]}")
print(f" -> Metadata Dims: {meta_vector.shape[1]}")
print(f" -> Text Dims:     {text_vector.shape[1]}")
print(f" -> Image Dims:    {image_vector.shape[1]}")

if X_final.shape[1] > 500:
    print("⚠️  NOTE: Dimensions > 500.")

# Save
x_path = os.path.join(OUTPUT_FOLDER, 'X_final.npy')
y_path = os.path.join(OUTPUT_FOLDER, 'y_final.npy')

np.save(x_path, X_final)
np.save(y_path, y_final)

print(f"\n✅ SUCCESS! Files saved to:")
print(f"   {x_path}")
print(f"   {y_path}")