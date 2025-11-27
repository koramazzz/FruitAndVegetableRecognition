import pandas as pd
import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import color
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
METADATA_PATH = '../dataset/raw/metadata.csv'
IMAGES_ORIGINAL_PATH = '../dataset/images/original'
IMAGES_GENERATED_PATH = '../dataset/images/generated'

# Feature extraction parameters
COLOR_HIST_BINS = 64  # Bins per channel for color histogram
LBP_RADIUS = 3
LBP_N_POINTS = LBP_RADIUS * 8

print("--- 1. LOADING DATA ---")
# Load metadata to get IDs and labels
df_meta = pd.read_csv(METADATA_PATH)
print(f"Loaded {len(df_meta)} samples from metadata")
print(f"Sample IDs: {df_meta['ID'].head().tolist()}")
print("-" * 50)


print("\n--- 2. EXTRACTING IMAGE FEATURES ---")

def extract_color_histogram(image, bins=COLOR_HIST_BINS):
    """
    Extract color histogram from RGB image.
    Returns: Flattened histogram vector (bins * 3 channels)
    """
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    # Normalize histograms
    hist_r = hist_r / (hist_r.sum() + 1e-7)
    hist_g = hist_g / (hist_g.sum() + 1e-7)
    hist_b = hist_b / (hist_b.sum() + 1e-7)
    
    # Flatten and concatenate
    color_features = np.hstack([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
    return color_features


def extract_lbp_features(image_gray):
    """
    Extract Local Binary Pattern (LBP) features.
    Returns: Uniform LBP histogram (59 dimensions)
    """
    # Calculate LBP
    lbp = local_binary_pattern(image_gray, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    
    # Calculate histogram (uniform patterns have 59 bins: 0-58)
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    return hist


def extract_glcm_features(image_gray):
    """
    Extract Gray-Level Co-occurrence Matrix (GLCM) features.
    Returns: 13 features (4 properties averaged + 9 individual angle features)
    """
    # Calculate GLCM for multiple angles
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                       levels=256, symmetric=True, normed=True)
    
    # Extract properties
    properties = ['contrast', 'energy', 'homogeneity', 'correlation']
    all_features = []
    
    # 1. Average features across all angles (4 dimensions)
    for prop in properties:
        prop_values = graycoprops(glcm, prop)
        all_features.append(np.mean(prop_values))
    
    # 2. Individual features for first 3 angles and first 3 properties (9 dimensions)
    # This gives us rotation-invariant + some directional information
    for angle_idx in range(3):  # Use first 3 angles
        for prop_idx in range(3):  # Use first 3 properties
            prop_values = graycoprops(glcm, properties[prop_idx])
            all_features.append(prop_values[0, angle_idx])
    
    # Total: 4 + 9 = 13 dimensions
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


def extract_all_features(image_path):
    """
    Extract all features from a single image.
    Returns: Combined feature vector
    """
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


def find_image_path(sample_id, label):
    """
    Find image path by checking both original and generated folders.
    Original folder may have _result suffix, generated folder has direct name.
    """
    label_lower = label.lower()
    
    # Try original folder first (may have _result suffix)
    original_paths = [
        os.path.join(IMAGES_ORIGINAL_PATH, label_lower, f"{sample_id}_result.jpg"),
        os.path.join(IMAGES_ORIGINAL_PATH, label_lower, f"{sample_id}.jpg"),
    ]
    
    for path in original_paths:
        if os.path.exists(path):
            return path
    
    # Try generated folder (category_gen subfolder)
    generated_path = os.path.join(IMAGES_GENERATED_PATH, f"{label_lower}_gen", f"{sample_id}.jpg")
    if os.path.exists(generated_path):
        return generated_path
    
    # Not found
    return None


# Extract features for all images
image_features_list = []
failed_ids = []
feature_dim = None  # Will be set after first successful extraction

print(f"Processing {len(df_meta)} images...")
for idx, row in df_meta.iterrows():
    sample_id = row['ID']
    label = row['label']
    
    # Find image path (checks both original and generated folders)
    image_path = find_image_path(sample_id, label)
    
    if image_path is None:
        print(f"  Image not found for {sample_id}")
        failed_ids.append(sample_id)
        # Use zero vector as placeholder
        if feature_dim is not None:
            image_features_list.append(np.zeros(feature_dim))
        continue
    
    try:
        features = extract_all_features(image_path)
        
        # Set feature dimension from first successful extraction
        if feature_dim is None:
            feature_dim = len(features)
        
        image_features_list.append(features)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df_meta)} images...")
            
    except Exception as e:
        print(f"  Error processing {sample_id}: {e}")
        failed_ids.append(sample_id)
        # Use zero vector as placeholder (dimension will be set from first successful)
        if feature_dim is not None:
            image_features_list.append(np.zeros(feature_dim))
        else:
            # If first image fails, skip it (will be handled later)
            pass

# Convert to numpy array
image_vector = np.array(image_features_list)

print(f"\nSuccessfully processed {len(image_features_list) - len(failed_ids)}/{len(df_meta)} images")
if failed_ids:
    print(f"Failed IDs: {failed_ids}")


print("\n--- 3. FEATURE DIMENSIONS ---")
print(f"Color Histogram: {COLOR_HIST_BINS * 3} dimensions")
print(f"LBP Features: {59} dimensions")
print(f"GLCM Features: {13} dimensions")
print(f"Statistical Features: {21} dimensions")
print(f"\nTotal Image Feature Vector Shape: {image_vector.shape}")
print(f"(Samples: {image_vector.shape[0]}, Dimensions: {image_vector.shape[1]})")
print(f"Expected Total: {COLOR_HIST_BINS * 3 + 59 + 13 + 21} dimensions")


print("\n--- 4. NORMALIZATION (OPTIONAL) ---")
# Optional: Normalize features using MinMaxScaler
scaler = MinMaxScaler()
image_vector_normalized = scaler.fit_transform(image_vector)
print(f"Normalized feature vector shape: {image_vector_normalized.shape}")

# Print sample feature vector (first 10 dimensions)
print("\nSample Feature Vector (First Sample, First 10 dimensions):")
print(np.round(image_vector_normalized[0][:10], 4))

