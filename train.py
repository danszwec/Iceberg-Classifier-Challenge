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
# Import the custom model and data processing
from sar_model import SARClassifier 
from data_prep import get_processed_data, AugmentedDataset, get_k_fold_data

# --- Hyperparameters ---
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 50



def plot_loss_and_accuracy(loss_train_history, accuracy_train_history, loss_val_history, accuracy_val_history, save_path=None):
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

  

# --- Training Function ---
def train_model(x_train, x_val, x_angle_train, x_angle_val, y_train, y_val):
    # 1. Load and Process Data (using your custom normalization)
    # Create PyTorch Datasets with augmentation for training
    train_dataset = AugmentedDataset(x_train, x_angle_train, y_train, augment=True)
    val_dataset = AugmentedDataset(x_val, x_angle_val, y_val, augment=False)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize Model and Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SARClassifier().to(device)
    
    # Use Binary Cross-Entropy Loss for binary classification
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Starting training on device: {device} ---\n")

    # 3. Training Loop
    loss_val_history = []
    accuracy_val_history = []
    loss_train_history = []
    accuracy_train_history = []
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, angles, targets) in enumerate(train_loader):
            images, angles, targets = images.to(device), angles.to(device), targets.to(device)

            # Forward pass
            outputs = model(images, angles)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_train_history.append(epoch_loss)
        
        # Calculate train accuracy
        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for images, angles, targets in train_loader:
                images, angles, targets = images.to(device), angles.to(device), targets.to(device)
                outputs = model(images, angles)
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
        train_accuracy = train_correct / train_total
        accuracy_train_history.append(train_accuracy)
        
        # 4. Validation Step
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, angles, targets in val_loader:
                images, angles, targets = images.to(device), angles.to(device), targets.to(device)
                
                outputs = model(images, angles)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_predictions / total_samples
        loss_val_history.append(val_loss)
        accuracy_val_history.append(val_accuracy)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    # Plot after training completes
    plot_loss_and_accuracy(loss_train_history, accuracy_train_history, loss_val_history, accuracy_val_history)

    print("\n--- Training Finished ---")
    #save the model
    torch.save(model.state_dict(), "sar_model.pth")


def k_fold_cross_validation(n_splits=5):
    """
    Perform k-fold cross-validation training.
    
    Args:
        n_splits: Number of folds (default: 5)
    """
    

    print(f"\n--- Starting {n_splits}-Fold Cross-Validation ---\n")
    
    # Store results for each fold
    fold_results = []
    all_val_losses = []
    all_val_accuracies = []
    
    # Iterate through folds
    for fold_idx, (x_train, x_val, x_angle_train, x_angle_val, y_train, y_val) in enumerate(get_k_fold_data(n_splits)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'='*60}\n")
        
        # Create datasets
        train_dataset = AugmentedDataset(x_train, x_angle_train, y_train, augment=True)
        val_dataset = AugmentedDataset(x_val, x_angle_val, y_val, augment=False)
        
        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model for this fold
        model = SARClassifier()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training history for this fold
        loss_train_history = []
        accuracy_train_history = []
        loss_val_history = []
        accuracy_val_history = []
        
        # Training loop
        for epoch in tqdm(range(NUM_EPOCHS), desc=f"Fold {fold_idx+1} Training"):
            model.train()
            running_loss = 0.0
            
            for batch_idx, (images, angles, targets) in enumerate(train_loader):
                images, angles, targets = images, angles, targets
                
                # Forward pass
                outputs = model(images, angles)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            loss_train_history.append(epoch_loss)
            
            # Calculate train accuracy
            model.eval()
            train_correct = 0
            train_total = 0
            with torch.no_grad():
                for images, angles, targets in train_loader:
                    images, angles, targets = images, angles, targets
                    outputs = model(images, angles)
                    predicted = (outputs > 0.5).float()
                    train_correct += (predicted == targets).sum().item()
                    train_total += targets.size(0)
            train_accuracy = train_correct / train_total
            accuracy_train_history.append(train_accuracy)
            
            # Validation step
            model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for images, angles, targets in val_loader:
                    images, angles, targets = images, angles, targets
                    
                    outputs = model(images, angles)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * images.size(0)
                    
                    predicted = (outputs > 0.5).float()
                    correct_predictions += (predicted == targets).sum().item()
                    total_samples += targets.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_accuracy = correct_predictions / total_samples
            loss_val_history.append(val_loss)
            accuracy_val_history.append(val_accuracy)
            
            if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Store fold results
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
        
        # Plot for this fold

        plot_loss_and_accuracy(loss_train_history, accuracy_train_history, 
                              loss_val_history, accuracy_val_history,save_path=f"k_fold_cross_validation/plots/loss_and_accuracy_fold_{fold_idx+1}.png")
       
        
        # Save model for this fold
        os.makedirs(f"k_fold_cross_validation/models", exist_ok=True)
        torch.save(model.state_dict(), f"k_fold_cross_validation/models/sar_model_fold_{fold_idx+1}.pth")
        print(f"\n✓ Fold {fold_idx+1} model saved as 'k_fold_cross_validation/models/sar_model_fold_{fold_idx+1}.pth'")
    
    # Print summary
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}\n")
    
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Val Loss: {result['best_val_loss']:.4f}, "
              f"Best Val Acc: {result['best_val_acc']:.4f}")
    
    avg_best_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
    avg_best_val_acc = np.mean([r['best_val_acc'] for r in fold_results])
    std_best_val_loss = np.std([r['best_val_loss'] for r in fold_results])
    std_best_val_acc = np.std([r['best_val_acc'] for r in fold_results])
    
    print(f"\nAverage Best Val Loss: {avg_best_val_loss:.4f} ± {std_best_val_loss:.4f}")
    print(f"Average Best Val Accuracy: {avg_best_val_acc:.4f} ± {std_best_val_acc:.4f}")
    print(f"\n--- K-Fold Cross-Validation Complete ---\n")
    
if __name__ == '__main__':
    # Choose training mode: 'kfold' for k-fold cross-validation, 'standard' for regular train/val split
    TRAINING_MODE = 'kfold'  # Change to 'standard' for regular training
    
    if TRAINING_MODE == 'kfold':
        # Run k-fold cross-validation
        k_fold_cross_validation(n_splits=5)
    else:
        # Standard train/val split training
        x_train, x_val, x_angle_train, x_angle_val, y_train, y_val = get_processed_data()
        train_model(x_train, x_val, x_angle_train, x_angle_val, y_train, y_val)