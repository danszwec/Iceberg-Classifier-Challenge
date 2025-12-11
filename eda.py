import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Tuple, List

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the data from a JSON file into a Pandas DataFrame."""
    try:
        data = json.load(open(file_path))
        df = pd.DataFrame(data)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        exit()


# Check integrity of all arrays simultaneously using a generator
def check_array(arr_list):
    try:
        arr = np.array(arr_list, dtype=np.float32)
        return arr.size == 5625 and not np.isnan(arr).any() and not np.isinf(arr).any()
    except:
        return False
        

def check_data_validity(train_df: pd.DataFrame,test_df: pd.DataFrame) -> bool:
    """
    Checks core data validity (bands and target) in the shortest form possible.
    Returns True if valid, False otherwise.
    args:
        df: pd.DataFrame
    returns:
        bool
    """
    # List to collect all issues
    issues: List[str] = []
    
    # 1. Check Target Column (is_iceberg)
    if 'is_iceberg' in train_df.columns:
        y = train_df['is_iceberg'].dropna().values
        if y.size != len(train_df):
            issues.append("Target has NaN/Null values.")
        if not np.all(np.isin(y, [0, 1])):
            issues.append("Target contains values other than 0 or 1.")

    for df in [train_df,test_df]:
        # 2. Check Input Bands (band_1, band_2)
        for band_name in ['band_1', 'band_2']:
            if band_name not in df.columns:
                issues.append(f"Missing column: '{band_name}'.")
                continue
                
        # Check for NaN/None in the Series itself
        if df[band_name].isnull().any() or df[band_name].map(lambda x: x is None).any():
            issues.append(f"Band '{band_name}' Series contains Null entries.")
            continue

        if not all(check_array(arr) for arr in df[band_name]):
            issues.append(f"Band '{band_name}' has incorrect size (not 75x75), non-numeric data, or NaN/Inf inside an array.")

        # Check inc_angle column for string 'na' values
        if 'inc_angle' in df.columns:
            na_string_count = df['inc_angle'].astype(str).str.lower().eq('na').sum()
            if na_string_count > 0:
                df['inc_angle'] = df['inc_angle'].replace(['na', 'NA', 'Na', 'nA'], None)
                issues.append(f"inc_angle column contains {na_string_count} string 'na' value(s) that should be converted to NaN.")

        #check there no dups in the data
        if df['id'].duplicated().any():
            issues.append("Data contains duplicates in the id column.")

        # Create a Series of the hashable combined band keys (Image data).
        band_keys = df.apply(lambda r: (tuple(r['band_1']), tuple(r['band_2'])), axis=1)

        # Group by the key and count unique IDs. Print if any key is linked to more than one unique ID.
        if (count := band_keys.groupby(band_keys).nunique().gt(1).sum()) > 0:
            issues.append(f"Data Redundancy Detected: {count} unique image key(s) are shared by multiple unique IDs.")

    # Check if the set of test IDs is different from the set of train IDs.
    if set(train_df['id']) == set(test_df['id']):
        issues.append("The test data and the train data have the same id values in the id column.")


    # Prepare comparable sets: Combine bands into a unique, hashable key (tuple of tuples).
    train_keys = set(train_df.apply(lambda r: (tuple(r['band_1']), tuple(r['band_2'])), axis=1))

    # Check test rows against the train keys and print the count of duplicates found.
    if (overlap_count := test_df.apply(lambda r: (tuple(r['band_1']), tuple(r['band_2'])) in train_keys, axis=1).sum()) > 0:
        issues.append(f"Data Leakage Detected: {overlap_count} test instance(s) are exact duplicates of a train instance.")

    # 3. Final Report
    if issues:
        print("VALIDITY CHECK FAILED.")
        for issue in issues:
            print(f" - {issue}")
        return False
    else:
        print("VALIDITY CHECK PASSED.")
        return True

def check_class_balance(df: pd.DataFrame) -> Tuple[dict, bool]:
    """
    Check if the target class is balanced.
    
    Args:
        df: DataFrame with 'is_iceberg' column
        
    Returns:
        Tuple containing (class_counts dict, is_balanced bool)
    """
    class_counts = df['is_iceberg'].value_counts().to_dict()
    total = len(df)
    percentages = {k: (v / total) * 100 for k, v in class_counts.items()}
    
    print("\n--- Class Balance Analysis ---")
    print(f"Total samples: {total}")
    print(f"Class 0 (ship): {class_counts.get(0, 0)} ({percentages.get(0, 0):.2f}%)")
    print(f"Class 1 (iceberg): {class_counts.get(1, 0)} ({percentages.get(1, 0):.2f}%)")
    
    # Consider balanced if both classes are within 40-60% range
    min_percentage = min(percentages.values())
    is_balanced = min_percentage >= 40.0
    
    if is_balanced:
        print("Data is relatively balanced")
    else:
        print("Data is imbalanced - consider using class weights or resampling")
    
    return class_counts, is_balanced

def check_angle_correlation(df: pd.DataFrame) -> float:
    """
    Check correlation between incidence angle and target.
    
    Args:
        df: DataFrame with 'inc_angle' and 'is_iceberg' columns
        
    Returns:
        Correlation coefficient
    """
    # Handle missing values
    df_clean = df.copy()
    df_clean['inc_angle'] = pd.to_numeric(df_clean['inc_angle'], errors='coerce')
    median_angle = df_clean['inc_angle'].median()
    df_clean['inc_angle'] = df_clean['inc_angle'].fillna(median_angle)
    
    correlation = df_clean['inc_angle'].corr(df_clean['is_iceberg'])
    
    print("\n--- Angle-Target Correlation Analysis ---")
    print(f"Correlation coefficient: {correlation:.4f}")
    
    if abs(correlation) < 0.1:
        print("Very weak correlation")
    elif abs(correlation) < 0.3:
        print("Weak correlation")
    elif abs(correlation) < 0.5:
        print("Moderate correlation")
    else:
        print("Strong correlation")
    
    return correlation

def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Check for missing values in the dataset.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with missing value counts
    """
    print("\n--- Missing Values Analysis ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    for col in df.columns:
        if missing[col] > 0:
            print(f"{col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        else:
            print(f"{col}: No missing values")
    
    return missing.to_dict()

def analyze_band_statistics(train_df: pd.DataFrame,test_df: pd.DataFrame) -> dict:
    """
    Analyze band statistics by class.
    
    Args:
        df: DataFrame with 'band_1', 'band_2', and 'is_iceberg' columns
        
    Returns:
        Dictionary with statistics
    """
    print("\n--- Band Statistics by Class ---")
    
    stats = {}
    for band_name in ['band_1', 'band_2']:
        print(f"\n{band_name}:")
        for class_label in [0, 1]:
            class_name = 'Ship' if class_label == 0 else 'Iceberg'
            band_values = np.concatenate([np.array(band) for band in train_df[train_df['is_iceberg'] == class_label][band_name]])
            mean_val = np.mean(band_values)
            std_val = np.std(band_values)
            min_val = np.min(band_values)
            max_val = np.max(band_values)
            
            print(f"  {class_name}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
            stats[f"{band_name}_class_{class_label}"] = {
                'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val
            }
    
    #check that the test data has the same band statistics as the train data
    for band_name in ['band_1', 'band_2']:
        band_values = np.concatenate([np.array(band) for band in test_df[band_name]])
        mean_val = np.mean(band_values)
        std_val = np.std(band_values)
        min_val = np.min(band_values)
        max_val = np.max(band_values)
        print(f"test {band_name}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
    
    return stats

def check_feature_correlation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Check correlation between band_1 and band_2.
    
    Args:
        train_df: DataFrame with 'band_1' and 'band_2' columns
        test_df: DataFrame with 'band_1' and 'band_2' columns
        
    Returns:
        None
    """
    print("\n--- Band Correlation Analysis ---")
    for i,df in enumerate([train_df, test_df]):
        band1_all = np.concatenate([np.array(band) for band in df['band_1']])
        band2_all = np.concatenate([np.array(band) for band in df['band_2']])
        correlation = np.corrcoef(band1_all, band2_all)[0, 1]

        if i == 0:
            print(f"train Correlation between band_1 and band_2: {correlation:.4f}")
        else:
            print(f"test Correlation between band_1 and band_2: {correlation:.4f}")
        #check the correlation between the bands and angles
        angle_series = pd.to_numeric(df['inc_angle'], errors='coerce')
        median_angle = angle_series.median()
        angle_all = angle_series.fillna(median_angle).values
        band1_means = np.array([np.mean(np.array(band)) for band in df['band_1']])
        band2_means = np.array([np.mean(np.array(band)) for band in df['band_2']])
        correlation_band1 = np.corrcoef(angle_all, band1_means)[0, 1]
        correlation_band2 = np.corrcoef(angle_all, band2_means)[0, 1]
        if i == 0:
            print(f"train Correlation between angle and band_1: {correlation_band1:.4f}")
            print(f"train Correlation between angle and band_2: {correlation_band2:.4f}")
        else:
            print(f"test Correlation between angle and band_1: {correlation_band1:.4f}")
            print(f"test Correlation between angle and band_2: {correlation_band2:.4f}")


def check_target_correlation(df: pd.DataFrame) -> None:
    """
    Check correlation between target and bands.
    
    Args:
        df: DataFrame with 'is_iceberg', 'band_1', and 'band_2' columns
        
    Returns:
        None
    """
    print("\n--- Target Correlation Analysis ---")
    target_series = df['is_iceberg']
    band1_means = np.array([np.mean(np.array(band)) for band in df['band_1']])
    band2_means = np.array([np.mean(np.array(band)) for band in df['band_2']])
    correlation_target_band1 = np.corrcoef(target_series, band1_means)[0, 1]
    correlation_target_band2 = np.corrcoef(target_series, band2_means)[0, 1]
    print(f"Correlation between target and band_1: {correlation_target_band1:.4f}")
    print(f"Correlation between target and band_2: {correlation_target_band2:.4f}")
    
def visualize_eda(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Create visualizations for EDA."""
    fig = plt.figure(figsize=(20, 10))
    
    # Class distribution
    ax1 = plt.subplot(2, 4, 1)
    class_counts = train_df['is_iceberg'].value_counts()
    ax1.bar(['Ship (0)', 'Iceberg (1)'], [class_counts.get(0, 0), class_counts.get(1, 0)], 
            color=['blue', 'orange'])
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Angle distribution by class
    ax2 = plt.subplot(2, 4, 2)
    df_clean = train_df.copy()
    df_clean['inc_angle'] = pd.to_numeric(df_clean['inc_angle'], errors='coerce')
    median_angle = df_clean['inc_angle'].median()
    df_clean['inc_angle'] = df_clean['inc_angle'].fillna(median_angle)
    
    ax2.hist(df_clean[df_clean['is_iceberg'] == 0]['inc_angle'], 
            bins=30, alpha=0.6, label='Ship (0)', color='blue')
    ax2.hist(df_clean[df_clean['is_iceberg'] == 1]['inc_angle'], 
            bins=30, alpha=0.6, label='Iceberg (1)', color='orange')
    ax2.set_xlabel('Incidence Angle')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Incidence Angle Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Band 1 distribution by class
    ax3 = plt.subplot(2, 4, 3)
    band1_ship = np.concatenate([np.array(band) for band in df_clean[df_clean['is_iceberg'] == 0]['band_1']])
    band1_iceberg = np.concatenate([np.array(band) for band in df_clean[df_clean['is_iceberg'] == 1]['band_1']])
    ax3.hist(band1_ship, bins=50, alpha=0.6, label='Ship (0)', color='blue', density=True)
    ax3.hist(band1_iceberg, bins=50, alpha=0.6, label='Iceberg (1)', color='orange', density=True)
    ax3.set_xlabel('Band 1 Value (dB)')
    ax3.set_ylabel('Density')
    ax3.set_title('Band 1 Distribution by Class')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Band 2 distribution by class
    ax4 = plt.subplot(2, 4, 4)
    band2_ship = np.concatenate([np.array(band) for band in df_clean[df_clean['is_iceberg'] == 0]['band_2']])
    band2_iceberg = np.concatenate([np.array(band) for band in df_clean[df_clean['is_iceberg'] == 1]['band_2']])
    ax4.hist(band2_ship, bins=50, alpha=0.6, label='Ship (0)', color='blue', density=True)
    ax4.hist(band2_iceberg, bins=50, alpha=0.6, label='Iceberg (1)', color='orange', density=True)
    ax4.set_xlabel('Band 2 Value (dB)')
    ax4.set_ylabel('Density')
    ax4.set_title('Band 2 Distribution by Class')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Band statistics box plot
    ax5 = plt.subplot(2, 4, 5)
    band1_means = np.array([np.mean(np.array(band)) for band in train_df['band_1']])
    band2_means = np.array([np.mean(np.array(band)) for band in train_df['band_2']])
    ship_mask = df_clean['is_iceberg'] == 0
    iceberg_mask = df_clean['is_iceberg'] == 1
    
    box_data = [band1_means[ship_mask], band1_means[iceberg_mask], 
                band2_means[ship_mask], band2_means[iceberg_mask]]
    ax5.boxplot(box_data, tick_labels=['B1 Ship', 'B1 Iceberg', 'B2 Ship', 'B2 Iceberg'])
    ax5.set_ylabel('Mean Band Value (dB)')
    ax5.set_title('Mean Band Values by Class')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    #compare the test and train angle distribution
    ax6 = plt.subplot(2, 4, 6)
    train_angle = df_clean['inc_angle'].values
    test_angle = pd.to_numeric(test_df['inc_angle'], errors='coerce').fillna(median_angle).values
    ax6.hist(train_angle, bins=30, alpha=0.6, label='Train', color="green")
    ax6.hist(test_angle, bins=30, alpha=0.6, label='Test', color='red')
    ax6.set_xlabel('Incidence Angle')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Incidence Angle Distribution')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)

    #compare the test and train band 1 distribution
    ax7 = plt.subplot(2, 4, 7)
    train_band1 = np.concatenate([np.array(band) for band in df_clean['band_1']])
    test_band1 = np.concatenate([np.array(band) for band in test_df['band_1']])
    ax7.hist(train_band1, bins=50, alpha=0.6, label='Train', color="green", density=True)
    ax7.hist(test_band1, bins=50, alpha=0.6, label='Test', color='red', density=True)
    ax7.set_xlabel('Band 1 Value (dB)')
    ax7.set_ylabel('Density')
    ax7.set_title('Band 1 Distribution')
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    
    #compare the test and train band 2 distribution
    ax8 = plt.subplot(2, 4, 8)
    train_band2 = np.concatenate([np.array(band) for band in df_clean['band_2']])
    test_band2 = np.concatenate([np.array(band) for band in test_df['band_2']])
    ax8.hist(train_band2, bins=50, alpha=0.6, label='Train', color="green", density=True)
    ax8.hist(test_band2, bins=50, alpha=0.6, label='Test', color='red', density=True)
    ax8.set_xlabel('Band 2 Value (dB)')
    ax8.set_ylabel('Density')
    ax8.set_title('Band 2 Distribution')
    ax8.legend(loc='upper right')
    ax8.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=150)
    print("\n✓ Visualization saved as 'eda_analysis.png'")
    plt.close()

def sample_images(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Plots sample SAR images from the train and test data for visual comparison.
    Shows one Ship and one Iceberg from the train set, and two arbitrary images from the test set.
    """
    
    plt.figure(figsize=(12, 5)) # Use a dedicated figure size for clarity

    # --- 1. Train Set Samples (Ship vs. Iceberg) ---
    
    # Safely get the index of the first Ship and first Iceberg
    ship_idx_list = train_df[train_df['is_iceberg'] == 0].index
    iceberg_idx_list = train_df[train_df['is_iceberg'] == 1].index
    
    # Ensure indices exist before trying to access them
    if not ship_idx_list.empty and not iceberg_idx_list.empty:
        ship_idx = ship_idx_list[0]
        iceberg_idx = iceberg_idx_list[0]
        
        # Reshape Band 1 vectors into 75x75 images
        ship_img = np.array(train_df.loc[ship_idx, 'band_1']).reshape(75, 75)
        iceberg_img = np.array(train_df.loc[iceberg_idx, 'band_1']).reshape(75, 75)
        
        combined_train = np.hstack([ship_img, iceberg_img])
        
        ax1 = plt.subplot(1, 2, 1) # Position 1 for train samples
        im = ax1.imshow(combined_train, cmap='viridis')
        ax1.set_title('Train Sample: Ship (left) vs Iceberg (right)')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    else:
        print("Warning: Train set does not contain both Ship and Iceberg samples.")


    # --- 2. Test Set Samples (Two arbitrary samples) ---
    
    if not test_df.empty:
        # We assume the test set does NOT have the 'is_iceberg' label (standard competition format)
        test_idx_1 = test_df.index[0]
        test_idx_2 = test_df.index[1]
        
        test_img_1 = np.array(test_df.loc[test_idx_1, 'band_1']).reshape(75, 75)
        test_img_2 = np.array(test_df.loc[test_idx_2, 'band_1']).reshape(75, 75)
        
        combined_test = np.hstack([test_img_1, test_img_2])
        
        ax2 = plt.subplot(1, 2, 2) # Position 2 for test samples
        im_test = ax2.imshow(combined_test, cmap='viridis')
        ax2.set_title(f"Test Samples: {test_df.loc[test_idx_1, 'id']} (left) vs {test_df.loc[test_idx_2, 'id']} (right)")
        ax2.axis('off')
        plt.colorbar(im_test, ax=ax2, fraction=0.046, pad=0.04)
    else:
        print("Warning: Test set is empty.")

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    print("\n✓ Sample images saved as 'sample_images.png' showing Train and Test examples.")
    plt.close()

def run_eda():
    """Main EDA function."""
    print("=== Exploratory Data Analysis ===")
    
    # Load data
    train_df = load_data('Data/train.json')
    test_df = load_data('Data/test.json')

    print("Checking data validity...")
    #check that all the values in the data is valid
    check_data_validity(train_df,test_df)
    
    print("Checking class balance...")
    # Check class balance
    check_class_balance(train_df)
    

    print("Checking band statistics...")
    # Analyze band statistics
    analyze_band_statistics(train_df,test_df)
    
    print("Checking feature correlation...")
    # Check band correlation
    check_feature_correlation(train_df,test_df)
    
    print("Checking target correlation...")
    # Check angle correlation
    check_target_correlation(train_df)
    
    Visualize
    visualize_eda(train_df, test_df)
    sample_images(train_df, test_df)
    print("\n=== EDA Complete ===")

if __name__ == '__main__':
    run_eda()

