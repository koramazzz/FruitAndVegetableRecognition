import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DESCRIPTION_PATH = '../dataset/description_raw.csv'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # Output dimension: 384

# 1. Load Data
try:
    df = pd.read_csv(DESCRIPTION_PATH)
except FileNotFoundError:
    print(f"Error: The file {DESCRIPTION_PATH} was not found. Please ensure the path is correct.")
    exit()

print("Loaded Descriptions (first 5 rows):")
print(df.head())
print("-" * 50)


# 2. Load the Sentence Transformer Model
print(f"Loading Sentence Transformer model: {EMBEDDING_MODEL}...")
try:
    # Download the model if not cached
    text_model = SentenceTransformer(EMBEDDING_MODEL) 
except Exception as e:
    print(f"Error loading model. Ensure 'sentence-transformers' is installed. Error: {e}")
    exit()

# 3. Generate text embeddings
# Convert the 'description' column to a list for encoding
descriptions = df['description'].tolist()
print(f"Encoding {len(descriptions)} sentences...")

# Convert all sentences into vectors of size 384
text_features = text_model.encode(descriptions, convert_to_numpy=True) 

# 4. Print the results
num_samples = text_features.shape[0]
num_dims = text_features.shape[1]

print(f"\nText Feature Shape: ({num_samples}, {num_dims})")
print(f"Dimensionality Check: {num_dims}. (Matches 384)")

# Print a snippet of the resulting vector
print("\nExample Text Feature Vector (First Sample, First 10 dimensions):")
print(np.round(text_features[0][:10], 4)) # Rounding for readability