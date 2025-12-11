# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import os
import yaml
# Import the custom model and data processing
from sar_model import SARClassifier 
from data_prep import get_processed_data, AugmentedDataset, get_k_fold_data, separate_test_data




def plot_loss_and_accuracy(loss_train_history: list, accuracy_train_history: list, loss_val_history: list, accuracy_val_history: list, save_path: str = None) -> None:
    """
    Plot the loss and accuracy curves.
    Args:
        loss_train_history: List of training losses
        accuracy_train_history: List of training accuracies
        loss_val_history: List of validation losses
        accuracy_val_history: List of validation accuracies
        save_path: Path to save the plot (default: None)
    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Loss
    ax1.plot(range(len(loss_train_history)), loss_train_history, 'b-', label="Train Loss")
    ax1.plot(range(len(loss_val_history)), loss_val_history, 'orange', label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(range(len(accuracy_train_history)), accuracy_train_history, 'b-', label="Train Accuracy")
    ax2.plot(range(len(accuracy_val_history)), accuracy_val_history, 'orange', label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.savefig("loss_and_accuracy.png")
        plt.close()

  
def train_model(x_train: torch.Tensor, x_val: torch.Tensor, y_train: torch.Tensor, y_val: torch.Tensor, fold_idx: int = None) -> tuple[list, list]:
    """
    Train the model on the given data.
    args:
        x_train: Tensor of shape (N, 2, 75, 75)
        x_val: Tensor of shape (N, 2, 75, 75)
        y_train: Tensor of shape (N, 1) containing the labels for the training data
        y_val: Tensor of shape (N, 1) containing the labels for the validation data
        fold_idx: Index of the fold (default: None)
    returns:
        loss_val_history: List of validation losses for each epoch
        accuracy_val_history: List of validation accuracies for each epoch
    """

    # 1. Create data sets and data loaders
    train_dataset = AugmentedDataset(x_train, y_train, augment=True)
    val_dataset = AugmentedDataset(x_val, y_val, augment=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #2. Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SARClassifier().to(device)
    

    # 3. Initialize the loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Initialize the training and validation history
    loss_val_history = []
    accuracy_val_history = []
    loss_train_history = []
    accuracy_train_history = []
    best_val_acc = 0.0

    print(f"\n--- Starting training on device: {device} ---\n")
    # 5. Training Loop
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        model.train()
        running_loss = 0.0
        
        # 5.1. run on each batch
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the running loss
            running_loss += loss.item() * images.size(0)

        # Update the epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)

        # Update the training history
        loss_train_history.append(epoch_loss)
        
        # Calculate train accuracy
        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
        train_accuracy = train_correct / train_total
        accuracy_train_history.append(train_accuracy)
        
        # 6. Validation Step
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)

                # Forward pass
                outputs = model(images)
                # Calculate the loss
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                # Calculate the accuracy
                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        #7.1Update the validation loss and accuracy
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_predictions / total_samples

        #7.2 Update the validation history
        loss_val_history.append(val_loss)
        accuracy_val_history.append(val_accuracy)

        #7.3 Print the training and validation metrics
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        #7.4 Save the weights with the highest validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            os.makedirs(f"k_fold_cross_validation/models/sar_model_fold_{fold_idx+1}", exist_ok=True)
            torch.save(model.state_dict(), f"k_fold_cross_validation/models/sar_model_fold_{fold_idx+1}/best_val_acc.pth")
            print(f"✓ Model saved as 'k_fold_cross_validation/models/sar_model_fold_{fold_idx+1}/best_val_acc.pth'")

    #8. Plot after training completes
    plot_loss_and_accuracy(loss_train_history, accuracy_train_history, loss_val_history, accuracy_val_history, save_path=f"k_fold_cross_validation/plots/loss_and_accuracy_fold_{fold_idx+1}.png")

    print("\n--- Training Finished ---")
    
    #9. Save the last model weights
    torch.save(model.state_dict(), f"k_fold_cross_validation/models/sar_model_fold_{fold_idx+1}/last.pth")

    #10. Return the validation loss and accuracy history
    return loss_val_history, accuracy_val_history


def k_fold_training_loop(data_path: str, n_splits: int, random_seed: int) -> None:
    """
    Perform k-fold training loop.
    args:
        data_path: Path to the data
        n_splits: Number of folds
        random_seed: Random seed
    returns:
        None
    """
    

    print(f"\n--- Starting {n_splits}-Fold Cross-Validation ---\n")
    #1. Store results for each fold
    fold_results = []
    all_val_losses = []
    all_val_accuracies = []
    
    #2. Iterate through folds
    for fold_idx, (x_train, x_val, y_train, y_val) in enumerate(get_k_fold_data(data_path, n_splits, random_seed)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}\n")
        
        #3. Train the model
        loss_val_history, accuracy_val_history = train_model(x_train, x_val, y_train, y_val, fold_idx)

        #4. Store fold results
        best_val_loss = min(loss_val_history)
        best_val_acc = max(accuracy_val_history)
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'final_val_loss': loss_val_history[-1],
            'final_val_acc': accuracy_val_history[-1]
        })
        all_val_losses.append(loss_val_history)
        all_val_accuracies.append(accuracy_val_history)
        
        
    #5. 1. Print summary
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    
    #5. 2. Print summary for each fold
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Val Loss: {result['best_val_loss']:.4f}, "
              f"Best Val Acc: {result['best_val_acc']:.4f}")
    
    #5. 3. Calculate the average and standard deviation of the best validation loss and accuracy
    avg_best_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    avg_best_val_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_best_val_loss = np.std([r['best_val_loss'] for r in fold_results])
    std_best_val_acc = np.std([r['best_val_acc'] for r in fold_results])
    
    #5. 4. Print the average and standard deviation of the best validation loss and accuracy
    print(f"\nAverage Best Val Loss: {avg_best_val_loss:.4f} ± {std_best_val_loss:.4f}")
    print(f"Average Best Val Accuracy: {avg_best_val_acc:.4f} ± {std_best_val_acc:.4f}")

    #5. 5. Print the completion message
    print(f"\n--- K-Fold Cross-Validation Complete ---\n")
    
if __name__ == '__main__':
    # Choose training mode: 'kfold' for k-fold cross-validation, 'standard' for regular train/val split
    with open('cfg.yaml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # --- Configuration ---
    TRAINING_MODE = cfg['training_mode']
    N_SPLITS = cfg['n_splits']
    RANDOM_SEED = cfg['random_seed']
    TEST_SIZE = cfg['test_size']
    LEARNING_RATE = cfg['learning_rate']
    BATCH_SIZE = cfg['batch_size']
    NUM_EPOCHS = cfg['num_epochs']
    DATA_PATH = cfg['data_path']

    #1. Separate the test data if the test size is greater than 0
    if TEST_SIZE > 0:
        separate_test_data(data_path=DATA_PATH, test_size=TEST_SIZE, random_seed=RANDOM_SEED)
        DATA_PATH = 'Data/new_train.json'
    
    #2. Run the training loop
    if TRAINING_MODE == 'kfold':
        # Run k-fold cross-validation
        k_fold_training_loop(data_path=DATA_PATH, n_splits=N_SPLITS, random_seed=RANDOM_SEED)

    else:
        # Standard train/val split training
        x_train, x_val, x_angle_train, x_angle_val, y_train, y_val = get_processed_data()
        train_model(x_train, x_val, x_angle_train, x_angle_val, y_train, y_val)