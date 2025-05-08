import matplotlib.pyplot as plt
import numpy as np
import re
import json
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def parse_args():
    """
    Parse command line arguments
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Visualize PSO Optimization Process')
    
    parser.add_argument('--results_file', type=str, default='pso_results.txt',
                        help='Path to PSO results file')
    parser.add_argument('--hyperparams_file', type=str, default='best_hyperparams.json',
                        help='Path to best hyperparameters file')
    
    return parser.parse_args()

def parse_pso_results(file_path):
    """
    Parse PSO results from file
    Args:
        file_path (str): Path to PSO results file
    Returns:
        iterations (list): List of iteration numbers
        accuracies (list): List of best accuracies
        hyperparams (list): List of best hyperparameters
    """
    iterations = []
    accuracies = []
    hyperparams = []
    
    if not os.path.exists(file_path):
        print(f"Results file not found: {file_path}")
        return iterations, accuracies, hyperparams
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract iteration data
    iteration_blocks = content.split('Iteration ')[1:]
    
    for block in iteration_blocks:
        # Extract iteration number
        iteration_match = re.match(r'(\d+):', block)
        if iteration_match:
            iteration = int(iteration_match.group(1))
            iterations.append(iteration)
            
            # Extract accuracy
            accuracy_match = re.search(r'Best accuracy: ([\d.]+)%', block)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
                accuracies.append(accuracy)
            
            # Extract hyperparameters
            hyperparams_match = re.search(r'Best hyperparameters: ({.*})', block)
            if hyperparams_match:
                try:
                    hyperparams_str = hyperparams_match.group(1)
                    # Replace single quotes with double quotes for JSON parsing
                    hyperparams_str = hyperparams_str.replace("'", '"')
                    hyperparams_dict = json.loads(hyperparams_str)
                    hyperparams.append(hyperparams_dict)
                except json.JSONDecodeError:
                    print(f"Error parsing hyperparameters in iteration {iteration}")
    
    return iterations, accuracies, hyperparams

def plot_accuracy_progress(iterations, accuracies, save_path='accuracy_progress.png'):
    """
    Plot accuracy progress over iterations
    Args:
        iterations (list): List of iteration numbers
        accuracies (list): List of best accuracies
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
    
    # Add best accuracy annotation
    best_idx = np.argmax(accuracies)
    best_iter = iterations[best_idx]
    best_acc = accuracies[best_idx]
    plt.annotate(f'Best: {best_acc:.2f}%',
                 xy=(best_iter, best_acc),
                 xytext=(best_iter, best_acc - 5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)
    
    plt.title('PSO Optimization Progress', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Best Accuracy (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(iterations)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Accuracy progress plot saved to {save_path}')

def plot_hyperparameter_evolution(iterations, hyperparams, save_path='hyperparameter_evolution.png'):
    """
    Plot hyperparameter evolution over iterations
    Args:
        iterations (list): List of iteration numbers
        hyperparams (list): List of best hyperparameters
        save_path (str): Path to save the plot
    """
    if not hyperparams:
        print("No hyperparameter data available")
        return
    
    # Extract hyperparameter values
    param_names = list(hyperparams[0].keys())
    param_values = {name: [] for name in param_names}
    
    for params in hyperparams:
        for name in param_names:
            param_values[name].append(params.get(name, 0))
    
    # Plot hyperparameter evolution
    plt.figure(figsize=(12, 8))
    
    for i, name in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        plt.plot(iterations, param_values[name], 'o-', linewidth=2, markersize=6)
        plt.title(f'{name.replace("_", " ").title()} Evolution', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(iterations)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Hyperparameter evolution plot saved to {save_path}')

def plot_hyperparameter_correlation(hyperparams, accuracies, save_path='hyperparameter_correlation.png'):
    """
    Plot correlation between hyperparameters and accuracy
    Args:
        hyperparams (list): List of best hyperparameters
        accuracies (list): List of best accuracies
        save_path (str): Path to save the plot
    """
    if not hyperparams or not accuracies:
        print("No hyperparameter or accuracy data available")
        return
    
    # Extract hyperparameter values
    param_names = list(hyperparams[0].keys())
    param_values = {name: [] for name in param_names}
    
    for params in hyperparams:
        for name in param_names:
            param_values[name].append(params.get(name, 0))
    
    # Plot correlations
    plt.figure(figsize=(12, 8))
    
    for i, name in enumerate(param_names):
        plt.subplot(2, 2, i+1)
        plt.scatter(param_values[name], accuracies, alpha=0.7, s=80)
        
        # Add trend line
        z = np.polyfit(param_values[name], accuracies, 1)
        p = np.poly1d(z)
        plt.plot(sorted(param_values[name]), p(sorted(param_values[name])), 'r--', linewidth=2)
        
        plt.title(f'Accuracy vs {name.replace("_", " ").title()}', fontsize=14)
        plt.xlabel(name.replace("_", " ").title(), fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'Hyperparameter correlation plot saved to {save_path}')

def plot_3d_hyperparameter_space(hyperparams, accuracies, save_path='hyperparameter_space_3d.png'):
    """
    Plot 3D visualization of hyperparameter space
    Args:
        hyperparams (list): List of best hyperparameters
        accuracies (list): List of best accuracies
        save_path (str): Path to save the plot
    """
    if not hyperparams or not accuracies or len(hyperparams) < 3:
        print("Not enough data for 3D visualization")
        return
    
    # Extract hyperparameter values
    param_names = list(hyperparams[0].keys())
    if len(param_names) < 3:
        print("Need at least 3 hyperparameters for 3D visualization")
        return
    
    # Select 3 hyperparameters for visualization
    selected_params = param_names[:3]
    x = [params[selected_params[0]] for params in hyperparams]
    y = [params[selected_params[1]] for params in hyperparams]
    z = [params[selected_params[2]] for params in hyperparams]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with accuracy as color
    scatter = ax.scatter(x, y, z, c=accuracies, cmap='viridis', s=100, alpha=0.7)
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Accuracy (%)', fontsize=12)
    
    # Set labels
    ax.set_xlabel(selected_params[0].replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(selected_params[1].replace("_", " ").title(), fontsize=12)
    ax.set_zlabel(selected_params[2].replace("_", " ").title(), fontsize=12)
    
    plt.title('3D Hyperparameter Space', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f'3D hyperparameter space plot saved to {save_path}')

def main():
    """
    Main function to visualize PSO optimization process
    """
    # Parse arguments
    args = parse_args()
    
    # Parse PSO results
    iterations, accuracies, hyperparams = parse_pso_results(args.results_file)
    
    if not iterations:
        print("No PSO results found. Run the PSO optimization first.")
        return
    
    # Plot accuracy progress
    plot_accuracy_progress(iterations, accuracies)
    
    # Plot hyperparameter evolution
    plot_hyperparameter_evolution(iterations, hyperparams)
    
    # Plot hyperparameter correlation
    plot_hyperparameter_correlation(hyperparams, accuracies)
    
    # Plot 3D hyperparameter space
    plot_3d_hyperparameter_space(hyperparams, accuracies)
    
    # Load best hyperparameters
    if os.path.exists(args.hyperparams_file):
        with open(args.hyperparams_file, 'r') as f:
            best_hyperparams = json.load(f)
        
        print("\n=== Best Hyperparameters ===")
        for param, value in best_hyperparams.items():
            print(f"{param}: {value}")
    
    print("\nVisualization complete. Check the output files for plots.")

if __name__ == "__main__":
    main()
