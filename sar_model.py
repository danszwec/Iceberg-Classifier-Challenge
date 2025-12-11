import torch
import torch.nn as nn
import torch.nn.functional as F

class SARClassifier(nn.Module):
    """
    Implements a dual-input CNN based on the simple architecture:
    2x (Conv + ReLU + Pool) -> Flatten -> Concatenate Angle -> Dense Layer -> Sigmoid Output.
    """
    def __init__(self):
        super(SARClassifier, self).__init__()
        
        # --- 1. Image Processing Branch (2x Conv + Pool) ---
        # Input: (N, 2, 75, 75)
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size after Conv1: 71x71. After Pool1: 35x35 (rounded down)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output size after Conv2: 31x31. After Pool2: 15x15 (rounded down)
        
        # Calculate the size of the flattened CNN output: 15 * 15 * 64 = 14400
        # The total input size for the Dense layer will be 14400 (from CNN) + 1 (from angle).
        cnn_output_features = 15 * 15 * 64 
        total_dense_input = cnn_output_features
        
        # --- 2. Fully Connected Layer (D) ---
        # We choose 512 as a typical size for the hidden layer "D"
        self.fc1 = nn.Linear(total_dense_input, 512)
        
        # --- 3. Output/Softmax Layer ---
        # Binary classification output: 1 neuron with Sigmoid activation
        self.fc_out = nn.Linear(512, 1)

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SARClassifier model.
        
        Args:
            x_image: Tensor of shape (N, 2, 75, 75) representing the SAR image.
            
        Returns:
            output: Tensor of shape (N, 1) representing the predicted probability of the image being an iceberg.
        """
        # 1. Process Image
        x_image = F.relu(self.conv1(x_image))
        x_image = self.pool1(x_image)
        
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        
        # Flatten the CNN output: (N, C*H*W)
        x_image = x_image.view(x_image.size(0), -1)
        
        # 2. Fully Connected Layers (D)
        x_image = F.relu(self.fc1(x_image))
        
        # 4. Output Layer (Sigmoid for binary classification probability)
        output = torch.sigmoid(self.fc_out(x_image))
        
        return output