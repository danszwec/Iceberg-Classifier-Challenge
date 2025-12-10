import pandas as pd
import numpy as np
import json
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
TEST_SIZE = 0.2
RANDOM_SEED = 42

def load_data(file_path):
    """Loads the data from a JSON file into a Pandas DataFrame."""
    try:
        data = json.load(open(file_path))
        df = pd.DataFrame(data)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure 'train.json' is in the current directory.")
        exit()

def N_x_normalization(band_data_db):
    """
    Applies the user-defined custom normalization N(x).
    
    This function first converts the dB input to Linear Power Intensity (x) 
    to correctly apply the threshold x >= 1.
    """
    
    # 1. CRITICAL CONVERSION: dB to Linear Power Intensity (x = 10^(dB / 10))
    x_linear = np.power(10, band_data_db / 10)
    
    # 2. Apply the piecewise function L(x)
    mask_greater_equal_one = x_linear >= 1
    L_x = np.zeros_like(x_linear)
    
    # Rule 1 (x < 1): L(x) = x
    L_x[~mask_greater_equal_one] = x_linear[~mask_greater_equal_one]
    
    # Rule 2 (x >= 1): L(x) = 1 + log(x) (using natural logarithm)
    L_x[mask_greater_equal_one] = 1 + np.log(x_linear[mask_greater_equal_one])
    
    # 3. Final Normalization: N(x) = L(x) / max(L(x))
    max_L_x = L_x.max()
    
    if max_L_x == 0:
        # Avoid division by zero, though highly unlikely
        return L_x
    
    N_x = L_x / max_L_x
    
    return N_x


def preprocess_images_two_channel(df):
    """
    Reshapes bands, applies custom normalization, and returns 
    a 2-channel tensor (N, C, H, W).
    """
    images = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        # Get raw dB data (input to the normalization function)
        band_1_raw = np.array(row['band_1']).reshape(75, 75)
        band_2_raw = np.array(row['band_2']).reshape(75, 75)

        # Apply custom normalization to each band independently
        band_1_norm = N_x_normalization(band_1_raw)
        band_2_norm = N_x_normalization(band_2_raw)

        # Stack channels: (75, 75, 2)
        image = np.dstack((band_1_norm, band_2_norm))
        
        images.append(image)

    X = np.array(images)
    
    # Transpose from (N, H, W, C) to (N, C, H, W) for PyTorch
    X = np.transpose(X, (0, 3, 1, 2))
    
    return X


def handle_metadata(df):
    """Handles the 'inc_angle' metadata feature and prepares it for the model (Min-Max scaling)."""
    
    # Impute and Min-Max scale incidence angle
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    median_angle = df['inc_angle'].median()
    df['inc_angle'] = df['inc_angle'].fillna(median_angle)
    
    min_val = df['inc_angle'].min()
    max_val = df['inc_angle'].max()
    X_angle = (df['inc_angle'] - min_val) / (max_val - min_val)
    
    return X_angle.values.reshape(-1, 1)

def check_class_distribution(y_train, y_val, y_full):
    """
    Check and print class distribution in train, validation, and full dataset.
    
    Args:
        y_train: Training labels
        y_val: Validation labels
        y_full: Full dataset labels
    """
    print("\n--- Class Distribution Check ---")
    
    # Full dataset
    full_counts = np.bincount(y_full.astype(int))
    full_total = len(y_full)
    full_pct = (full_counts / full_total) * 100
    
    # Train dataset
    train_counts = np.bincount(y_train.astype(int))
    train_total = len(y_train)
    train_pct = (train_counts / train_total) * 100
    
    # Validation dataset
    val_counts = np.bincount(y_val.astype(int))
    val_total = len(y_val)
    val_pct = (val_counts / val_total) * 100
    
    print(f"Full Dataset: Class 0={full_counts[0]} ({full_pct[0]:.2f}%), Class 1={full_counts[1]} ({full_pct[1]:.2f}%)")
    print(f"Train Set:    Class 0={train_counts[0]} ({train_pct[0]:.2f}%), Class 1={train_counts[1]} ({train_pct[1]:.2f}%)")
    print(f"Val Set:      Class 0={val_counts[0]} ({val_pct[0]:.2f}%), Class 1={val_counts[1]} ({val_pct[1]:.2f}%)")
    
    # Check if distributions are similar (within 2% difference)
    train_diff = abs(train_pct[0] - full_pct[0])
    val_diff = abs(val_pct[0] - full_pct[0])
    
    if train_diff < 2.0 and val_diff < 2.0:
        print("✓ All subsets maintain balanced class distribution")
    else:
        print(f"⚠ Warning: Class distribution differs by more than 2%")
        print(f"  Train difference: {train_diff:.2f}%, Val difference: {val_diff:.2f}%")

# --- Main Execution ---
def get_processed_data():
    """Executes the full pipeline and returns PyTorch tensors, ready for training."""
    
    print("--- Starting Custom Two-Channel Data Processing ---")
    
    # 1. Load Data
    train_df = load_data('train.json')
    y = train_df['is_iceberg'].values

    # 2. Preprocess Image Data using custom normalization
    X_image = preprocess_images_two_channel(train_df)
    
    # 3. Handle Metadata
    X_angle = handle_metadata(train_df)

    # 4. Split Data
    X_img_train, X_img_val, X_angle_train, X_angle_val, y_train, y_val = \
        train_test_split(X_image, X_angle, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
    
    # Check class distribution in subsets
    check_class_distribution(y_train, y_val, y)

    # 5. Convert to PyTorch Tensors
    X_img_train_t = torch.from_numpy(X_img_train).float()
    X_img_val_t = torch.from_numpy(X_img_val).float()
    
    X_angle_train_t = torch.from_numpy(X_angle_train).float()
    X_angle_val_t = torch.from_numpy(X_angle_val).float()
    
    y_train_t = torch.from_numpy(y_train).float().view(-1, 1)
    y_val_t = torch.from_numpy(y_val).float().view(-1, 1)

    print("\n--- Data Preparation Complete ---")
    print(f"Train Image Tensor Shape: {X_img_train_t.shape} (N, C, H, W)")
    print(f"Validation Image Tensor Shape: {X_img_val_t.shape}")
    print(f"Image values are now non-linearly compressed (0 to 1 range).")
    
    return X_img_train_t, X_img_val_t, X_angle_train_t, X_angle_val_t, y_train_t, y_val_t

if __name__ == "__main__":
    get_processed_data()