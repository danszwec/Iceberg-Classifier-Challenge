import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the data from a JSON file into a Pandas DataFrame."""
    try:
        data = json.load(open(file_path))
        df = pd.DataFrame(data)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        exit()

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
    max_percentage = max(percentages.values())
    is_balanced = min_percentage >= 40.0
    
    if is_balanced:
        print("✓ Data is relatively balanced")
    else:
        print("⚠ Data is imbalanced - consider using class weights or resampling")
    
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

def analyze_band_statistics(df: pd.DataFrame) -> dict:
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
            band_values = np.concatenate([np.array(band) for band in df[df['is_iceberg'] == class_label][band_name]])
            mean_val = np.mean(band_values)
            std_val = np.std(band_values)
            min_val = np.min(band_values)
            max_val = np.max(band_values)
            
            print(f"  {class_name}: mean={mean_val:.2f}, std={std_val:.2f}, min={min_val:.2f}, max={max_val:.2f}")
            stats[f"{band_name}_class_{class_label}"] = {
                'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val
            }
    
    return stats

def check_band_correlation(df: pd.DataFrame) -> float:
    """
    Check correlation between band_1 and band_2.
    
    Args:
        df: DataFrame with 'band_1' and 'band_2' columns
        
    Returns:
        Correlation coefficient
    """
    print("\n--- Band Correlation Analysis ---")
    
    band1_all = np.concatenate([np.array(band) for band in df['band_1']])
    band2_all = np.concatenate([np.array(band) for band in df['band_2']])
    
    correlation = np.corrcoef(band1_all, band2_all)[0, 1]
    print(f"Correlation between band_1 and band_2: {correlation:.4f}")
    
    if abs(correlation) < 0.3:
        print("Low correlation - bands provide complementary information")
    elif abs(correlation) < 0.7:
        print("Moderate correlation")
    else:
        print("High correlation - bands may be redundant")
    
    return correlation

def visualize_eda(df: pd.DataFrame):
    """Create visualizations for EDA."""
    fig = plt.figure(figsize=(16, 10))
    
    # Class distribution
    ax1 = plt.subplot(2, 3, 1)
    class_counts = df['is_iceberg'].value_counts()
    ax1.bar(['Ship (0)', 'Iceberg (1)'], [class_counts.get(0, 0), class_counts.get(1, 0)], 
            color=['blue', 'orange'])
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Angle distribution by class
    ax2 = plt.subplot(2, 3, 2)
    df_clean = df.copy()
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
    ax3 = plt.subplot(2, 3, 3)
    band1_ship = np.concatenate([np.array(band) for band in df[df['is_iceberg'] == 0]['band_1']])
    band1_iceberg = np.concatenate([np.array(band) for band in df[df['is_iceberg'] == 1]['band_1']])
    ax3.hist(band1_ship, bins=50, alpha=0.6, label='Ship (0)', color='blue', density=True)
    ax3.hist(band1_iceberg, bins=50, alpha=0.6, label='Iceberg (1)', color='orange', density=True)
    ax3.set_xlabel('Band 1 Value (dB)')
    ax3.set_ylabel('Density')
    ax3.set_title('Band 1 Distribution by Class')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Band 2 distribution by class
    ax4 = plt.subplot(2, 3, 4)
    band2_ship = np.concatenate([np.array(band) for band in df[df['is_iceberg'] == 0]['band_2']])
    band2_iceberg = np.concatenate([np.array(band) for band in df[df['is_iceberg'] == 1]['band_2']])
    ax4.hist(band2_ship, bins=50, alpha=0.6, label='Ship (0)', color='blue', density=True)
    ax4.hist(band2_iceberg, bins=50, alpha=0.6, label='Iceberg (1)', color='orange', density=True)
    ax4.set_xlabel('Band 2 Value (dB)')
    ax4.set_ylabel('Density')
    ax4.set_title('Band 2 Distribution by Class')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Band statistics box plot
    ax5 = plt.subplot(2, 3, 5)
    band1_means = np.array([np.mean(np.array(band)) for band in df['band_1']])
    band2_means = np.array([np.mean(np.array(band)) for band in df['band_2']])
    ship_mask = df['is_iceberg'] == 0
    iceberg_mask = df['is_iceberg'] == 1
    
    box_data = [band1_means[ship_mask], band1_means[iceberg_mask], 
                band2_means[ship_mask], band2_means[iceberg_mask]]
    ax5.boxplot(box_data, tick_labels=['B1 Ship', 'B1 Iceberg', 'B2 Ship', 'B2 Iceberg'])
    ax5.set_ylabel('Mean Band Value (dB)')
    ax5.set_title('Mean Band Values by Class')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # Sample images
    ax6 = plt.subplot(2, 3, 6)
    ship_idx = df[df['is_iceberg'] == 0].index[0]
    iceberg_idx = df[df['is_iceberg'] == 1].index[0]
    
    ship_img = np.array(df.loc[ship_idx, 'band_1']).reshape(75, 75)
    iceberg_img = np.array(df.loc[iceberg_idx, 'band_1']).reshape(75, 75)
    
    combined = np.hstack([ship_img, iceberg_img])
    im = ax6.imshow(combined, cmap='viridis')
    ax6.set_title('Sample Images: Ship (left) vs Iceberg (right)')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=150)
    print("\n✓ Visualization saved as 'eda_analysis.png'")
    plt.close()

def run_eda():
    """Main EDA function."""
    print("=== Exploratory Data Analysis ===")
    
    # Load data
    df = load_data('train.json')
    
    # Check class balance
    class_counts, is_balanced = check_class_balance(df)
    
    # Check missing values
    missing = check_missing_values(df)
    
    # Analyze band statistics
    band_stats = analyze_band_statistics(df)
    
    # Check band correlation
    band_corr = check_band_correlation(df)
    
    # Check angle correlation
    angle_corr = check_angle_correlation(df)
    
    # Visualize
    visualize_eda(df)
    
    print("\n=== EDA Complete ===")

if __name__ == '__main__':
    run_eda()

