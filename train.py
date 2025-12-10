# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
# Import the custom model and data processing
from sar_model import SARClassifier 
from data_prep import get_processed_data 

# --- Hyperparameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20


def plot_loss_and_accuracy(loss_train_history, accuracy_train_history, loss_val_history, accuracy_val_history):
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
    
    plt.tight_layout()
    plt.savefig("loss_and_accuracy.png")
    plt.close()

# --- Training Function ---
def train_model():
    # 1. Load and Process Data (using your custom normalization)
    X_img_train, X_img_val, X_angle_train, X_angle_val, y_train, y_val = get_processed_data()

    # Create PyTorch Datasets
    train_dataset = TensorDataset(X_img_train, X_angle_train, y_train)
    val_dataset = TensorDataset(X_img_val, X_angle_val, y_val)

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

if __name__ == '__main__':
    # Make sure you have the files train.json, data_pipeline.py, and model.py 
    # in the same directory before running this.
    train_model()