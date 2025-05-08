import torch
import torch.nn as nn
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from model import get_model
from utils import get_dataloaders, evaluate_model
from tqdm import tqdm
import json

def parse_args():
    """
    Parse command line arguments
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Compare Original and PSO-Optimized Models')
    
    parser.add_argument('--train_dir', type=str, default='plant disease dataset/train',
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='plant disease dataset/validation',
                        help='Path to validation data directory')
    parser.add_argument('--test_dir', type=str, default='plant disease dataset/test',
                        help='Path to test data directory')
    
    parser.add_argument('--original_model', type=str, default='plant_disease_model.pt',
                        help='Path to original model weights')
    parser.add_argument('--optimized_model', type=str, default='pso_optimized_model.pt',
                        help='Path to PSO-optimized model weights')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    return parser.parse_args()

def get_predictions(model, dataloader, device):
    """
    Get predictions from a model on a dataset
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    Returns:
        all_predictions: List of predicted classes
        all_labels: List of true labels
        all_confidences: List of prediction confidences
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Getting predictions'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    return all_predictions, all_labels, all_confidences

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Plot confusion matrix
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Confusion matrix saved to {save_path}')

def plot_confidence_distribution(confidences_original, confidences_optimized, save_path):
    """
    Plot confidence distribution
    Args:
        confidences_original: Confidences from original model
        confidences_optimized: Confidences from optimized model
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Convert to percentages
    confidences_original = np.array(confidences_original) * 100
    confidences_optimized = np.array(confidences_optimized) * 100
    
    # Plot histograms
    plt.hist(confidences_original, bins=20, alpha=0.5, label='Original Model', color='blue')
    plt.hist(confidences_optimized, bins=20, alpha=0.5, label='PSO-Optimized Model', color='green')
    
    plt.title('Prediction Confidence Distribution', fontsize=16)
    plt.xlabel('Confidence (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Confidence distribution plot saved to {save_path}')

def plot_accuracy_comparison(accuracy_original, accuracy_optimized, save_path):
    """
    Plot accuracy comparison
    Args:
        accuracy_original: Accuracy of original model
        accuracy_optimized: Accuracy of optimized model
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    models = ['Original Model', 'PSO-Optimized Model']
    accuracies = [accuracy_original, accuracy_optimized]
    
    bars = plt.bar(models, accuracies, color=['blue', 'green'])
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
    
    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Accuracy comparison plot saved to {save_path}')

def main():
    """
    Main function to compare original and PSO-optimized models
    """
    # Parse arguments
    args = parse_args()
    
    # Check if model files exist
    if not os.path.exists(args.original_model):
        raise FileNotFoundError(f"Original model not found: {args.original_model}")
    if not os.path.exists(args.optimized_model):
        raise FileNotFoundError(f"Optimized model not found: {args.optimized_model}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    _, _, test_loader = get_dataloaders(
        args.train_dir, args.val_dir, args.test_dir, batch_size=args.batch_size
    )
    
    # Get class names
    test_dataset = test_loader.dataset
    class_names = test_dataset.classes
    
    # Load hyperparameters for optimized model
    if os.path.exists('best_hyperparams.json'):
        with open('best_hyperparams.json', 'r') as f:
            best_hyperparams = json.load(f)
        print(f"Loaded best hyperparameters: {best_hyperparams}")
    else:
        best_hyperparams = {'dropout_rate': 0.3}  # Default if file not found
        print("Best hyperparameters file not found, using default dropout rate of 0.3")
    
    # Initialize and load original model
    original_model = get_model(device=device)
    original_model.load_state_dict(torch.load(args.original_model, map_location=device))
    original_model.eval()
    
    # Initialize and load optimized model
    optimized_model = get_model(dropout_rate=best_hyperparams.get('dropout_rate', 0.3), device=device)
    optimized_model.load_state_dict(torch.load(args.optimized_model, map_location=device))
    optimized_model.eval()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate models
    print("\n=== Evaluating Original Model ===")
    original_accuracy, original_loss = evaluate_model(original_model, test_loader, criterion, device)
    print(f'Test Loss: {original_loss:.4f}, Test Accuracy: {original_accuracy:.2f}%')
    
    print("\n=== Evaluating PSO-Optimized Model ===")
    optimized_accuracy, optimized_loss = evaluate_model(optimized_model, test_loader, criterion, device)
    print(f'Test Loss: {optimized_loss:.4f}, Test Accuracy: {optimized_accuracy:.2f}%')
    
    # Get predictions
    print("\nGetting predictions from original model...")
    original_preds, original_labels, original_confidences = get_predictions(original_model, test_loader, device)
    
    print("Getting predictions from optimized model...")
    optimized_preds, optimized_labels, optimized_confidences = get_predictions(optimized_model, test_loader, device)
    
    # Plot confusion matrices
    plot_confusion_matrix(
        original_labels, original_preds, class_names,
        'Confusion Matrix - Original Model',
        'confusion_matrix_original.png'
    )
    
    plot_confusion_matrix(
        optimized_labels, optimized_preds, class_names,
        'Confusion Matrix - PSO-Optimized Model',
        'confusion_matrix_optimized.png'
    )
    
    # Plot confidence distribution
    plot_confidence_distribution(
        original_confidences, optimized_confidences,
        'confidence_distribution.png'
    )
    
    # Plot accuracy comparison
    plot_accuracy_comparison(
        original_accuracy, optimized_accuracy,
        'accuracy_comparison.png'
    )
    
    # Print classification reports
    print("\n=== Classification Report - Original Model ===")
    print(classification_report(original_labels, original_preds, target_names=class_names))
    
    print("\n=== Classification Report - PSO-Optimized Model ===")
    print(classification_report(optimized_labels, optimized_preds, target_names=class_names))
    
    # Calculate improvement
    accuracy_improvement = optimized_accuracy - original_accuracy
    confidence_improvement = (np.mean(optimized_confidences) - np.mean(original_confidences)) * 100
    
    print("\n=== Improvement Summary ===")
    print(f"Accuracy improvement: {accuracy_improvement:.2f}%")
    print(f"Average confidence improvement: {confidence_improvement:.2f}%")
    
    # Save results to file
    with open('model_comparison_results.txt', 'w') as f:
        f.write("=== Model Comparison Results ===\n\n")
        f.write(f"Original Model Accuracy: {original_accuracy:.2f}%\n")
        f.write(f"PSO-Optimized Model Accuracy: {optimized_accuracy:.2f}%\n")
        f.write(f"Accuracy Improvement: {accuracy_improvement:.2f}%\n\n")
        f.write(f"Original Model Average Confidence: {np.mean(original_confidences)*100:.2f}%\n")
        f.write(f"PSO-Optimized Model Average Confidence: {np.mean(optimized_confidences)*100:.2f}%\n")
        f.write(f"Confidence Improvement: {confidence_improvement:.2f}%\n\n")
        f.write(f"Best Hyperparameters: {best_hyperparams}\n")
    
    print("\nResults saved to model_comparison_results.txt")

if __name__ == "__main__":
    main()
