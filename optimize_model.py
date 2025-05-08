import os
import torch
import argparse
from pso_optimizer import PSOOptimizer
from model import get_model
from utils import get_dataloaders, evaluate_model
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json

def parse_args():
    """
    Parse command line arguments
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='PSO Hyperparameter Optimization for Plant Disease Detection')
    
    parser.add_argument('--train_dir', type=str, default='plant disease dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='plant disease dataset/validation',
                        help='Path to validation data directory')
    parser.add_argument('--test_dir', type=str, default='plant disease dataset/test',
                        help='Path to test data directory')
    
    parser.add_argument('--num_particles', type=int, default=10,
                        help='Number of particles in the swarm')
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='Maximum number of PSO iterations')
    parser.add_argument('--epochs_per_iteration', type=int, default=5,
                        help='Number of epochs to train each model during PSO')
    
    parser.add_argument('--final_epochs', type=int, default=20,
                        help='Number of epochs to train the final model')
    parser.add_argument('--model_save_path', type=str, default='pso_optimized_model.pt',
                        help='Path to save the optimized model')
    
    return parser.parse_args()

def train_final_model(hyperparams, train_dir, val_dir, test_dir, epochs, model_save_path):
    """
    Train the final model with the best hyperparameters found by PSO
    Args:
        hyperparams (dict): Best hyperparameters found by PSO
        train_dir (str): Path to training data
        val_dir (str): Path to validation data
        test_dir (str): Path to test data
        epochs (int): Number of epochs to train for
        model_save_path (str): Path to save the trained model
    Returns:
        test_accuracy (float): Test accuracy of the final model
        train_history (dict): Training history
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir, val_dir, test_dir, batch_size=hyperparams['batch_size']
    )
    
    # Initialize model
    model = get_model(dropout_rate=hyperparams['dropout_rate'], device=device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparams['learning_rate'], 
        weight_decay=hyperparams['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for images, labels in train_loader:
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
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation accuracy: {val_accuracy:.2f}%')
    
    # Load best model for testing
    model.load_state_dict(torch.load(model_save_path))
    
    # Evaluate on test set
    test_accuracy, test_loss = evaluate_model(model, test_loader, criterion, device)
    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    return test_accuracy, history

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    Args:
        history (dict): Training history
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Training history plot saved to {save_path}')

def main():
    """
    Main function to run PSO optimization and train the final model
    """
    # Parse arguments
    args = parse_args()
    
    # Check if directories exist
    for dir_path in [args.train_dir, args.val_dir, args.test_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Define hyperparameter bounds
    hyperparameter_bounds = {
        'learning_rate': (0.0001, 0.01),
        'batch_size': (16, 64),
        'dropout_rate': (0.1, 0.5),
        'weight_decay': (1e-6, 1e-3)
    }
    
    # Initialize PSO optimizer
    pso = PSOOptimizer(
        num_particles=args.num_particles,
        max_iterations=args.max_iterations,
        hyperparameter_bounds=hyperparameter_bounds,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir
    )
    
    # Run PSO optimization
    print("=== Starting PSO Optimization ===")
    best_hyperparams, best_fitness = pso.optimize(epochs_per_iteration=args.epochs_per_iteration)
    
    # Save best hyperparameters to file
    with open('best_hyperparams.json', 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    
    # Train final model with best hyperparameters
    print("\n=== Training Final Model with Best Hyperparameters ===")
    test_accuracy, history = train_final_model(
        hyperparams=best_hyperparams,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        epochs=args.final_epochs,
        model_save_path=args.model_save_path
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best validation accuracy during PSO: {best_fitness:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Optimized model saved to: {args.model_save_path}")

if __name__ == "__main__":
    main()
