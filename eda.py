import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

from data_prep import load_data

# Check integrity of all  simultaneously using a generator
def check_array(arr_list: list) -> bool:
    """
    Checks the integrity of an array.
    args:
        arr_list: list
    returns:
        bool
    """
    try:
        arr = np.array(arr_list, dtype=np.float32)
        return arr.size == 5625 and not np.isnan(arr).any() and not np.isinf(arr).any()
    except:
        return False
        

def check_data_validity(train_df: pd.DataFrame,test_df: pd.DataFrame) -> bool:
    """
    Checks core data validity (bands and target) in the shortest form possible.
    args:
        train_df: pd.DataFrame of training data
        test_df: pd.DataFrame of test data
    returns:
        bool: True if valid, False otherwise
    """
    # List to collect all issues
    issues = []
    
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

def check_class_balance(df: pd.DataFrame) -> None:
    """
    Check if the target class is balanced.
    
    args:
        df: pd.DataFrame with 'is_iceberg' column
    returns:
        None
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
    
    return None

def check_angle_target_correlation(df: pd.DataFrame) -> None:
    """
    Check correlation between incidence angle and target.
    
    args:
        df: pd.DataFrame with 'inc_angle' and 'is_iceberg' columns
    returns:
        None
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
    
    return None

def check_missing_values(df: pd.DataFrame) -> None:
    """
    Check for missing values in the dataset.
    
    args:
        df: DataFrame to check
    returns:
        None
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



def plot_angle_iceberg_histogram(df: pd.DataFrame, min_angle: float, max_angle: float) -> None:
    """
    Filter angles between 36-40, round to 4 decimals, and plot histogram
    showing angle distribution colored by is_iceberg.
    
    args:
        df: pd.DataFrame with 'inc_angle' and 'is_iceberg' columns
        min_angle: float: minimum angle to plot
        max_angle: float: maximum angle to plot
    returns:
        None
    """
    print("\n--- Angle-Iceberg Histogram Analysis (36-40 degrees) ---")
    
    # Convert angles to numeric and filter
    df_clean = df.copy()
    df_clean['inc_angle'] = pd.to_numeric(df_clean['inc_angle'], errors='coerce')
    
    # Filter angles between 36-40
    filtered_df = df_clean[(df_clean['inc_angle'] >= min_angle) & (df_clean['inc_angle'] <= max_angle)].copy()
    
    if len(filtered_df) == 0:
        print("No data found with angles between 36-40 degrees")
        return
    
    # Round to 4 decimal places
    filtered_df['inc_angle_rounded'] = filtered_df['inc_angle'].round(4)
    
    print(f"Found {len(filtered_df)} samples with angles between 36-40 degrees")
    print(f"  Ships (0): {len(filtered_df[filtered_df['is_iceberg'] == 0])}")
    print(f"  Icebergs (1): {len(filtered_df[filtered_df['is_iceberg'] == 1])}")
    
    # Create histogram plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Separate data by class
    ship_angles = filtered_df[filtered_df['is_iceberg'] == 0]['inc_angle_rounded']
    iceberg_angles = filtered_df[filtered_df['is_iceberg'] == 1]['inc_angle_rounded']
    
    # Get unique angle values (rounded to 4 decimals) for binning
    unique_angles = sorted(filtered_df['inc_angle_rounded'].unique())
    num_bins = min(len(unique_angles), 100)  # Limit to reasonable number of bins
    
    # Create histogram with appropriate binning
    ax.hist(ship_angles, bins=num_bins, alpha=0.6, label='Ship (0)', color='blue', density=False)
    ax.hist(iceberg_angles, bins=num_bins, alpha=0.6, label='Iceberg (1)', color='orange', density=False)
    
    ax.set_xlabel('Incidence Angle (rounded to 4 decimals)', fontsize=12)
    ax.set_ylabel('Count (is_iceberg)', fontsize=12)
    ax.set_title('Histogram of Angles (36-40°) by Class', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set more x-axis ticks to show more values
    min_angle = filtered_df['inc_angle_rounded'].min()
    max_angle = filtered_df['inc_angle_rounded'].max()
    
    # Create ticks every 0.1 degrees or use unique values if fewer
    if len(unique_angles) <= 50:
        # Show all unique angles if there aren't too many
        ax.set_xticks(unique_angles)
        ax.set_xticklabels([f'{angle:.4f}' for angle in unique_angles], rotation=45, ha='right', fontsize=8)
    else:
        # Show ticks at regular intervals
        num_ticks = min(50, len(unique_angles))
        tick_positions = np.linspace(min_angle, max_angle, num_ticks)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f'{angle:.4f}' for angle in tick_positions], rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('eda_plots/angle_iceberg_histogram.png', dpi=150)
    print("Visualization saved as 'eda_plots/angle_iceberg_histogram.png'")
    plt.close()
    return None

def visualize_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Create visualizations for EDA.
    args:
        train_df: pd.DataFrame of training data
        test_df: pd.DataFrame of test data
    returns:
        None
    """
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
    plt.savefig('eda_plots/eda_analysis.png', dpi=150)
    print("\n✓ Visualization saved as 'eda_plots/eda_analysis.png'")
    plt.close()

def sample_images(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Plots sample SAR images from the train and test data for visual comparison.
    Shows one Ship and one Iceberg from the train set, and two arbitrary images from the test set.
    """
    
    plt.figure(figsize=(12, 5)) # Use a dedicated figure size for clarity

    #1. Plot the train set samples (Ship vs. Iceberg)
    
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


    #2. Plot the test set samples (Two arbitrary samples)
    
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
    plt.savefig('eda_plots/sample_images.png', dpi=150)
    print("\n✓ Sample images saved as 'eda_plots/sample_images.png' showing Train and Test examples.")
    plt.close()
    return None

def run_eda(train_path: str, test_path: str) -> None:
    """
    Main EDA function.
    args:
        train_path: str
        test_path: str
    returns:
        None
    """
    print("=== Exploratory Data Analysis ===")
    
    #create the eda_plots directory if it doesn't exist
    os.makedirs('eda_plots', exist_ok=True)
    # Load data
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    print("Checking data validity...")
    #check that all the values in the data is valid
    check_data_validity(train_df,test_df)
    
    print("Checking class balance...")
    # Check class balance
    check_class_balance(train_df)
    
    print("Checking target correlation...")
    # Check the correlation between the incidence angle and the target
    check_angle_target_correlation(train_df)
    
    print("Visualizing the EDA...")
    # Visualize the EDA
    visualize_eda(train_df, test_df)
    sample_images(train_df, test_df)


    #zoom in on the angle distribution
    plot_angle_iceberg_histogram(train_df, 36, 40)
    print("\n=== EDA Complete ===")

if __name__ == '__main__':
    run_eda(train_path='Data/train.json', test_path='Data/test.json')

