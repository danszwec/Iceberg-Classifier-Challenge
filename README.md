# SAR Image Classification: Iceberg vs Ship Detection

## Project Report : A detailed explanation of my approach is provided in the accompanying PDF.

A deep learning project for classifying Synthetic Aperture Radar (SAR) images to distinguish between icebergs and ships using PyTorch. The project includes comprehensive data preprocessing, exploratory data analysis, k-fold cross-validation training, and evaluation capabilities.



## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Format](#data-format)


## Installation

### Using Docker (Recommended)

1. **Build the Docker image**:
```bash
docker build -t sar-classifier .
```

2. **Run the container**:
```bash
docker run -it --rm \
    -v $(pwd):/app \
    -v $(pwd)/Data:/app/Data \
    sar-classifier
```

Mounts:
- $(pwd):/app      # Mounts the project repo into the container
- $(pwd)/Data:/app/Data   # Ensures the Data directory is available inside the container
*/


3. **Ensure data files are in place**:
   - `Data/train.json` - Training data with labels
   - `Data/test.json` - Test data (without labels)

### Manual Installation (Alternative)

If you prefer not to use Docker:

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure data files are in place**:
   - `Data/train.json` - Training data with labels
   - `Data/test.json` - Test data (without labels)

## Project Structure

```
├── cfg.yaml                       # Configuration file
├── data_prep.py                   # Data preprocessing and loading
├── eda.py                         # Exploratory data analysis
├── train.py                       # Training script
├── evaluation.py                  # Model evaluation and predictions
├── sar_model.py                   # CNN model architecture
└── requirements.txt               # Python dependencies
```

##  Usage
**Make sure `cfg.yaml` is updated with your desired parameters before running any processing, training, or evaluation steps.**
- If you change parameters (e.g., validation size, random seed), re-run data and training scripts to apply the updated configuration.


### 1. Exploratory Data Analysis

Run EDA to understand your data:

```bash
python eda.py
```

This will generate:
- Class distribution analysis
- Band statistics by class
- Feature correlations
- Angle distribution visualizations
- Sample image visualizations
- Saved plots in the current directory

### 2. Training

#### K-Fold Cross-Validation (Recommended)

Edit `cfg.yaml` to set `training_mode: 'kfold'`, then run:

```bash
python train.py
```

This will:
- Split training data into train/test sets (if not already done)
- Perform k-fold cross-validation (default: 10 folds)
- Save model checkpoints for each fold
- Generate training plots for each fold
- Save models in `k_fold_cross_validation/models/`

#### Standard Train/Val Split

Edit `cfg.yaml` to set `training_mode: 'standard'`, then run:

```bash
python train.py
```

### 3. Evaluation

Evaluate models on the held-out test set:

```bash
python evaluation.py
```

This will:
- Evaluate all k-fold models on the test set
- Generate confusion matrices for each fold
- Select the best model based on log loss
- Generate submission predictions using the best model
- Save results in `k_fold_cross_validation/plots/`

## Configuration

Edit `cfg.yaml` to customize training parameters:

```yaml
# Data Configuration
validation_size: 0.1          # Validation split size (for standard mode)
random_seed: 42               # Random seed for reproducibility
test_size: 0.07               # Test set size (held-out from training)
data_path: 'Data/train.json'  # Path to training data

# Training Configuration
learning_rate: 0.00005        # Learning rate for optimizer
batch_size: 64               # Batch size
num_epochs: 50               # Number of training epochs
training_mode: 'kfold'       # 'kfold' or 'standard'
n_splits: 10                 # Number of folds for k-fold CV
```

## Data Format

### Input Data Structure

The JSON files should contain records with the following structure:

```json
[
  {
    "id": "image_id",
    "band_1": [5625 float values],  # 75x75 SAR image in dB
    "band_2": [5625 float values],  # 75x75 SAR image in dB
    "inc_angle": 36.5,              # Incidence angle (or "na")
    "is_iceberg": 1                 # 0 = ship, 1 = iceberg (only in train.json)
  }
]
```

## Output Files

### EDA Outputs
- `eda_analysis.png` - Comprehensive EDA visualizations
- `sample_images.png` - Sample SAR images from train/test sets
- `angle_iceberg_histogram.png` - Zoomed-in incidence angle histogram (36–40°) by class

### Training Outputs

- `k_fold_cross_validation/models/sar_model_fold_{N}/last.pth` - Final model checkpoint
- `k_fold_cross_validation/models/sar_model_fold_{N}/best_val_acc.pth` - Best validation accuracy checkpoint
- `k_fold_cross_validation/plots/loss_and_accuracy_fold_{N}.png` - Training curves

### Evaluation Outputs

- `k_fold_cross_validation/plots/confusion_matrix_fold_{N}.png` - Confusion matrices
- `submission.csv` - Final predictions for test set



