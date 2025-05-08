import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model import get_model
from utils import get_dataloaders, evaluate_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(train_dir, val_dir, test_dir, 
                batch_size=32, 
                learning_rate=0.001,
                num_epochs=20,  # Increased epochs
                dropout_rate=0.3,  # Reduced dropout
                model_save_path='plant_disease_model.pt'):
    """
    Train the plant disease detection model
    Args:
        train_dir (str): Path to training data
        val_dir (str): Path to validation data
        test_dir (str): Path to test data
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        num_epochs (int): Number of training epochs
        dropout_rate (float): Dropout rate for regularization
        model_save_path (str): Path to save the trained model
    """
    # Check if directories exist
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not os.listdir(dir_path):
            raise ValueError(f"Directory is empty: {dir_path}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Get data loaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_dir, val_dir, test_dir, batch_size=batch_size
        )
        
        # Initialize model
        model = get_model(dropout_rate=dropout_rate, device=device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        # Training loop
        best_val_accuracy = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training phase
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate training accuracy and loss
            train_accuracy = 100 * correct / total
            train_loss = running_loss / len(train_loader)
            
            # Validation phase
            val_accuracy, val_loss = evaluate_model(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(val_accuracy)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved with validation accuracy: {val_accuracy:.2f}%')
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_accuracy, test_loss = evaluate_model(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Dataset paths
        train_dir = "plant disease dataset/train"
        val_dir = "plant disease dataset/validation"
        test_dir = "plant disease dataset/test"
        
        # Training parameters
        params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 20,  # Increased epochs
            'dropout_rate': 0.3,  # Reduced dropout
            'model_save_path': 'plant_disease_model.pt'
        }
        
        # Start training
        train_model(train_dir, val_dir, test_dir, **params)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure:")
        print("1. The dataset directories exist and are properly organized")
        print("2. The dataset contains images in the correct format (jpg, jpeg, png)")
        print("3. You have sufficient disk space and memory")
        print("4. You have the required dependencies installed") 