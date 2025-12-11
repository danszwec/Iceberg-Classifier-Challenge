# evaluation.py
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.metrics import log_loss, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from tqdm import tqdm
from sar_model import SARClassifier 
from data_prep import load_data, get_test_data


def save_confusion_matrix(confusion_matrix: np.ndarray, save_path: str) -> None:
    """
    Save the confusion matrix as a heatmap.
    args:
        confusion_matrix: Confusion matrix
        save_path: Path to save the confusion matrix
    returns:
        None
    """

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    plt.colorbar()
    
    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, f'{confusion_matrix[i, j]:.2f}', 
                    ha='center', va='center', color='black' if confusion_matrix[i, j] < 0.5 else 'white')
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks([0, 1], ['0', '1'])
    plt.yticks([0, 1], ['0', '1'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return None

def evaluate(model_path: str, batch_size: int, X_img_test: torch.Tensor, X_angle_test: torch.Tensor, y_test: torch.Tensor) -> tuple[np.ndarray, float, float]:
    """
    Evaluate the model on the validation set and submit the predictions to the test set.
    args:
        model_path: Path to the model weights
        batch_size: Batch size for the validation set
        X_img_test: Test images
        X_angle_test: Test angles
        y_test: Test labels
    returns:
        tuple[np.ndarray, float, float]
    """
    #1. Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SARClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #2. Load the validation split data (includes labels)
    val_dataset = TensorDataset(X_img_test, y_test)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    #3.1. Predictions and targets
    val_predictions = []
    val_targets = []
    
    #3.2. Calculate the metrics on the validation set
    print("\n--- Evaluating Metrics on Validation Set ---")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            
            outputs = model(images)
            val_predictions.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(targets.cpu().numpy().flatten())
            
    val_targets = np.array(val_targets)
    val_predictions_proba = np.array(val_predictions)
    
    # 3.3Convert probabilities to binary predictions for classification metrics
    val_predictions_binary = (val_predictions_proba > 0.5).astype(int)

    #4. 1. Calculate the confusion matrix
    confusion_matrix = sklearn_confusion_matrix(val_targets, val_predictions_binary, labels=[0, 1], normalize='true')

    #4. 2. Calculate the log loss
    log_loss_test = log_loss(val_targets, val_predictions_proba)

    #4. 3. Calculate the F1 score
    f1_micro = f1_score(val_targets, val_predictions_binary, average='micro')

    #4. 4. Calculate the precision score
    precision_micro = precision_score(val_targets, val_predictions_binary, average='micro')

    #4. 5. Calculate the recall score
    recall_micro = recall_score(val_targets, val_predictions_binary, average='micro')
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"Precision Micro: {precision_micro:.4f}")
    print(f"Recall Micro: {recall_micro:.4f}")

    return confusion_matrix, f1_micro, log_loss_test
    
def submit_predictions(model_path: str, batch_size: int) -> None:
    """
    Submit the predictions to the test set.
    args:
        model_path: Path to the model weights
        batch_size: Batch size for the test set
    returns:
        None
    """
    #1. Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SARClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #2. Load the Test Set data (no labels)
    X_img_test, test_ids, _ = get_test_data(data_path='Data/test.json')

    test_dataset = TensorDataset(X_img_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    #3. Predictions
    test_predictions = []
    

    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            test_predictions.extend(outputs.cpu().numpy().flatten())

    #4. Create the Submission File
    submission_df = pd.DataFrame({
        'id': test_ids,
        'is_iceberg': test_predictions
    })
    
    submission_file_name = 'submission.csv'
    submission_df.to_csv(submission_file_name, index=False)
    
    print(f"\nSubmission file successfully created: {submission_file_name}")
    print("This file is ready to be uploaded for competition scoring.")
    return None 


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    N_SPLITS = cfg['n_splits']
    f1_micro_list = []
    log_loss_test_list = []

    #1. Load the test data
    X_img_test, X_angle_test, _, y_test = get_test_data(data_path='Data/new_train.json')

    #2. Evaluate the model on the test set
    for fold_idx in tqdm(range(N_SPLITS)):
        confusion_matrix, f1_micro, log_loss_test = evaluate(model_path=f'k_fold_cross_validation/models/sar_model_fold_{fold_idx+1}/last.pth', batch_size=64, X_img_test=X_img_test, X_angle_test=X_angle_test, y_test=y_test)
        f1_micro_list.append(f1_micro)
        log_loss_test_list.append(log_loss_test)
        save_confusion_matrix(confusion_matrix, f'k_fold_cross_validation/plots/confusion_matrix_fold_{fold_idx+1}.png')

    #3. Take the best performing model and submit the predictions
    best_fold_idx = np.argmax(log_loss_test_list)
    best_model_path = f'k_fold_cross_validation/models/sar_model_fold_{best_fold_idx+1}/best_val_acc.pth'

    #4. Submit the predictions
    print(f"Best model path: {best_model_path} with log loss: {log_loss_test_list[best_fold_idx]:.4f}")
    submit_predictions(model_path=best_model_path, batch_size=64)


    