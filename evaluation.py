# evaluation.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import log_loss, f1_score
import numpy as np

# Import the custom model and data processing functions
from sar_model import SARClassifier 
from data_prep import get_processed_data 

# --- Configuration ---
MODEL_PATH = 'sar_classifier_model.pth'
BATCH_SIZE = 64

def evaluate_and_submit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SARClassifier().to(device)
    
    try:
        # Load the trained model weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please run train.py first.")
        return

    # --- PART 1: Calculate Metrics on Validation Set ---
    
    print("\n--- 1. Evaluating Metrics on Validation Set ---")
    # Load the validation split data (includes labels)
    _, X_img_val, _, X_angle_val, _, y_val = get_processed_data()

    val_dataset = TensorDataset(X_img_val, X_angle_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for images, angles, targets in val_loader:
            images, angles = images.to(device), angles.to(device)
            
            outputs = model(images, angles)
            val_predictions.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(targets.cpu().numpy().flatten())
            
    val_targets = np.array(val_targets)
    val_predictions = np.array(val_predictions)

    # Calculate Log Loss (Binary Cross-Entropy Loss)
    # The Log Loss is a standard competition metric.
    logloss = log_loss(val_targets, val_predictions)
    
    # Calculate F1 Score (requires converting probabilities to binary predictions)
    # We choose a threshold of 0.5
    binary_predictions = (val_predictions > 0.5).astype(int)
    f1 = f1_score(val_targets, binary_predictions)

    print(f"Validation Log Loss: {logloss:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    
    
    # --- PART 2: Generate Submission on Test Set ---
    
    print("\n--- 2. Generating Predictions for Test Set ---")
    # Load the Test Set data (no labels)
    X_img_test, X_angle_test, test_ids = get_processed_data()

    test_dataset = TensorDataset(X_img_test, X_angle_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    test_predictions = []
    
    with torch.no_grad():
        for images, angles in test_loader:
            images, angles = images.to(device), angles.to(device)
            outputs = model(images, angles)
            test_predictions.extend(outputs.cpu().numpy().flatten())

    # Create the Submission File
    submission_df = pd.DataFrame({
        'id': test_ids,
        'is_iceberg': test_predictions
    })
    
    submission_file_name = 'submission.csv'
    submission_df.to_csv(submission_file_name, index=False)
    
    print(f"\nSubmission file successfully created: {submission_file_name}")
    print("This file is ready to be uploaded for competition scoring.")


if __name__ == '__main__':
    evaluate_and_submit()