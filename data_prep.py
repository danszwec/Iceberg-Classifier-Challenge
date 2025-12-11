import pandas as pd
import numpy as np
import json
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Generator
import random
import yaml

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the data from a JSON file into a Pandas DataFrame.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        data = json.load(open(file_path))
        df = pd.DataFrame(data)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        exit()

def N_x_normalization(band_data_db: np.ndarray) -> np.ndarray:
    """

    Applies the user-defined paper was given normalization N(x).
    
    This function first converts the dB input to Linear Power Intensity (x) 
    to correctly apply the threshold x >= 1.

    args:
        band_data_db: np.ndarray
    returns:
        np.ndarray
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


def preprocess_images_two_channel(df: pd.DataFrame) -> np.ndarray:
    """
    Reshapes bands, applies custom normalization, and returns 
    a 2-channel tensor (N, C, H, W).
    args:
        df: pd.DataFrame
    returns:
        np.ndarray
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


def handle_metadata(df: pd.DataFrame) -> np.ndarray:
    """
    Handles the 'inc_angle' metadata feature and prepares it for the model (Min-Max scaling).
    args:
        df: pd.DataFrame
    returns:
        np.ndarray
    """
    
    # Impute and Min-Max scale incidence angle
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    median_angle = df['inc_angle'].median()
    df['inc_angle'] = df['inc_angle'].fillna(median_angle)
    
    min_val = df['inc_angle'].min()
    max_val = df['inc_angle'].max()
    X_angle = (df['inc_angle'] - min_val) / (max_val - min_val)
    
    return X_angle.values.reshape(-1, 1)

def check_class_distribution(y_train: np.ndarray, y_val: np.ndarray, y_full: np.ndarray, subset_name: str = 'validation') -> None:
    """
    Check and print class distribution in train, validation, and full dataset.
    
    Args:
        y_train: np.ndarray of training labels
        y_val: np.ndarray of validation labels
        y_full: np.ndarray of full dataset labels
        subset_name: Name of the subset to check (default: 'validation')
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
    print(f"{subset_name} Set:      Class 0={val_counts[0]} ({val_pct[0]:.2f}%), Class 1={val_counts[1]} ({val_pct[1]:.2f}%)")
    
    # Check if distributions are similar (within 2% difference)
    train_diff = abs(train_pct[0] - full_pct[0])
    val_diff = abs(val_pct[0] - full_pct[0])
    
    if train_diff < 2.0 and val_diff < 2.0:
        print("All subsets maintain balanced class distribution")
    else:
        print(f"Warning: Class distribution differs by more than 2%")
        print(f"  Train difference: {train_diff:.2f}%, {subset_name} difference: {val_diff:.2f}%")

def separate_test_data(data_path: str, test_size: float, random_seed: int) -> None:
    """
    Separates test data from the training data.
    args:
        data_path: str
        test_size: float
        random_seed: int
    returns:
        None
    """

    train_df = load_data(data_path)
    y = train_df['is_iceberg'].values

    X_image = preprocess_images_two_channel(train_df)
    X_angle = handle_metadata(train_df)
    _, _, _, _, y_train, y_test = \
        train_test_split(X_image, X_angle, y, test_size=test_size, random_state=random_seed, stratify=y)

    # Check class distribution in subsets
    check_class_distribution(y_train, y_test, y, subset_name='test')

    # retrieve the test indices by comparing y_train to y (the original order)
    # test indices are those NOT in y_train after the split
    train_indices = set(np.where(np.in1d(y, y_train))[0])
    all_indices = set(range(len(y)))
    test_indices = list(all_indices - train_indices)

    # Drop the test data from train_df and y using test_indices
    train_df = train_df.drop(train_df.index[test_indices])
    test_df = train_df.iloc[test_indices]
    
    #write the test data and the train data to a json file
    train_df.to_json('Data/new_train.json', orient='records')
    test_df.to_json('Data/new_test.json', orient='records')


    return None

# --- Main Execution ---
def get_processed_data(data_path: str, validation_size: float = 0.1, random_seed: int = 42) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Executes the full pipeline and returns PyTorch tensors, ready for training.
    args:
        data_path: str
        validation_size: float
        random_seed: int
    returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    
    print("--- Starting Custom Two-Channel Data Processing ---")
    
    # 1. Load Data
    train_df = load_data(data_path)
    y = train_df['is_iceberg'].values

    # 2. Preprocess Image Data using custom normalization
    X_image = preprocess_images_two_channel(train_df)
    
    # 3. Handle Metadata
    X_angle = handle_metadata(train_df)

    
    # 4 split to train and validation
    X_img_train, X_img_val, X_angle_train, X_angle_val, y_train, y_val = \
        train_test_split(X_image, X_angle, y, test_size=validation_size, random_state=random_seed, stratify=y)
    
    # 5.Check class distribution in subsets
    check_class_distribution(y_train, y_val, y)

    # 6. Convert to PyTorch Tensors
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

def get_test_data(data_path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and process test data for predictions.
    args:
        data_path: str
    returns:
        tuple[torch.Tensor, torch.Tensor, list, torch.Tensor]
    """
    print("--- Processing Test Data ---")
    
    # Load test data
    test_df = load_data(data_path)
    
    # Handle 'id' column - use it if exists, otherwise use index
    if 'id' in test_df.columns:
        test_ids = test_df['id'].tolist()
    else:
        test_ids = test_df.index.tolist()

    # Preprocess images
    X_image_test = preprocess_images_two_channel(test_df)
    

    # Convert to PyTorch tensors
    X_img_test_t = torch.from_numpy(X_image_test).float()
    
    # Handle labels if they exist
    if 'is_iceberg' in test_df.columns:
        y_test = torch.from_numpy(test_df['is_iceberg'].values).float().view(-1, 1)
    else:
        y_test = None
    
    print(f"Test Image Tensor Shape: {X_img_test_t.shape}")
    
    return X_img_test_t, test_ids, y_test

class AugmentedDataset(Dataset):
    """
    Dataset wrapper that applies data augmentations during training.
    Supports horizontal/vertical flips and rotation.
    args:
        images: torch.Tensor
        labels: torch.Tensor
        augment: bool
    returns:
        image: torch.Tensor
        label: torch.Tensor
    """
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = True):
        self.images = images
        self.labels = labels
        self.augment = augment
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.augment:
            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])  # Flip along width
            
            # Random vertical flip (50% chance)
            if random.random() > 0.5:
                image = torch.flip(image, dims=[1])  # Flip along height
            
            # Random rotation (0, 90, 180, or 270 degrees)
            rotation = random.choice([0, 1, 2, 3])
            if rotation > 0:
                image = torch.rot90(image, k=rotation, dims=[1, 2])
        
        return image, label


def get_k_fold_data(data_path: str, n_splits: int, random_seed: int) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    Load and process data for k-fold cross-validation.
    args:
        data_path: str
        n_splits: int
        random_seed: int
    returns:
        Generator yielding tuples of (X_img_train, X_img_val, X_angle_train, X_angle_val, y_train, y_val) for each fold
    """
    print("--- Preparing K-Fold Cross-Validation Data ---")
    
    train_df = load_data(data_path)
    y = train_df['is_iceberg'].values
    
    # Preprocess all images and angles once
    X_image = preprocess_images_two_channel(train_df)
    
    # Use StratifiedKFold to maintain class balance
    k_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    for fold_idx, (train_index, val_index) in enumerate(k_folds.split(X_image, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        
        # Split data
        X_img_train = X_image[train_index]
        X_img_val = X_image[val_index]
        y_train = y[train_index]
        y_val = y[val_index]
        
        # Check class distribution
        check_class_distribution(y_train, y_val, y)
        
        # Convert to PyTorch tensors
        X_img_train_t = torch.from_numpy(X_img_train).float()
        X_img_val_t = torch.from_numpy(X_img_val).float()
        y_train_t = torch.from_numpy(y_train).float().view(-1, 1)
        y_val_t = torch.from_numpy(y_val).float().view(-1, 1)
        
        yield X_img_train_t, X_img_val_t, y_train_t, y_val_t


if __name__ == "__main__":

    #load the configuration file
    with open('cfg.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    #get the configuration values
    VALIDATION_SIZE = cfg['validation_size']
    TEST_SIZE = cfg['test_size']
    RANDOM_SEED = cfg['random_seed']
    DATA_PATH = cfg['data_path']    

    #separate the test data if the test size is greater than 0
    if TEST_SIZE > 0:
        separate_test_data(data_path=DATA_PATH, test_size=TEST_SIZE, random_seed=RANDOM_SEED)
        data_path = 'Data/new_train.json'
    else:
        data_path = DATA_PATH
    get_processed_data(data_path=data_path, validation_size=VALIDATION_SIZE, random_seed=RANDOM_SEED)