import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import random
import time
from model import get_model
from utils import get_dataloaders, evaluate_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Particle:
    """
    Represents a particle in the PSO algorithm
    """
    def __init__(self, bounds, dimensions):
        """
        Initialize a particle with random position and velocity
        Args:
            bounds (list): List of tuples (min, max) for each dimension
            dimensions (int): Number of dimensions (hyperparameters)
        """
        self.position = np.array([random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dimensions)])
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')  # For accuracy, higher is better
        self.current_fitness = float('-inf')
        
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """
        Update the velocity of the particle
        Args:
            global_best_position (np.array): Best position found by any particle
            w (float): Inertia weight
            c1 (float): Cognitive coefficient
            c2 (float): Social coefficient
        """
        r1 = np.random.random(len(self.position))
        r2 = np.random.random(len(self.position))
        
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
        
    def update_position(self, bounds):
        """
        Update the position of the particle and ensure it stays within bounds
        Args:
            bounds (list): List of tuples (min, max) for each dimension
        """
        self.position = self.position + self.velocity
        
        # Ensure position stays within bounds
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity

class PSOOptimizer:
    """
    PSO optimizer for hyperparameter tuning
    """
    def __init__(self, num_particles=10, max_iterations=20, 
                 hyperparameter_bounds=None, train_dir=None, val_dir=None, test_dir=None):
        """
        Initialize the PSO optimizer
        Args:
            num_particles (int): Number of particles in the swarm
            max_iterations (int): Maximum number of iterations
            hyperparameter_bounds (dict): Dictionary of hyperparameter bounds
            train_dir (str): Path to training data
            val_dir (str): Path to validation data
            test_dir (str): Path to test data
        """
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        
        # Default hyperparameter bounds if none provided
        if hyperparameter_bounds is None:
            self.hyperparameter_bounds = {
                'learning_rate': (0.0001, 0.01),
                'batch_size': (16, 64),
                'dropout_rate': (0.1, 0.5),
                'weight_decay': (1e-6, 1e-3)
            }
        else:
            self.hyperparameter_bounds = hyperparameter_bounds
            
        # Convert bounds to list for easier access
        self.bounds = [self.hyperparameter_bounds[key] for key in self.hyperparameter_bounds.keys()]
        self.dimensions = len(self.bounds)
        
        # Initialize particles
        self.particles = [Particle(self.bounds, self.dimensions) for _ in range(num_particles)]
        self.global_best_position = np.zeros(self.dimensions)
        self.global_best_fitness = float('-inf')
        
        # Data directories
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        
        # Hyperparameter keys
        self.param_keys = list(self.hyperparameter_bounds.keys())
        
    def evaluate_particle(self, particle, epochs=5):
        """
        Evaluate a particle by training a model with its hyperparameters
        Args:
            particle (Particle): Particle to evaluate
            epochs (int): Number of epochs to train for
        Returns:
            fitness (float): Fitness value (validation accuracy)
        """
        # Convert particle position to hyperparameters
        hyperparams = {}
        for i, key in enumerate(self.param_keys):
            if key == 'batch_size':
                hyperparams[key] = int(particle.position[i])
            else:
                hyperparams[key] = particle.position[i]
        
        # Print hyperparameters
        print(f"\nEvaluating hyperparameters: {hyperparams}")
        
        # Train and evaluate model
        val_accuracy = self.train_and_evaluate(
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            dropout_rate=hyperparams['dropout_rate'],
            weight_decay=hyperparams['weight_decay'],
            epochs=epochs
        )
        
        return val_accuracy
    
    def train_and_evaluate(self, learning_rate, batch_size, dropout_rate, weight_decay, epochs=5):
        """
        Train a model with the given hyperparameters and evaluate it
        Args:
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            dropout_rate (float): Dropout rate for regularization
            weight_decay (float): Weight decay for optimizer
            epochs (int): Number of epochs to train for
        Returns:
            val_accuracy (float): Validation accuracy
        """
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Get data loaders
            train_loader, val_loader, _ = get_dataloaders(
                self.train_dir, self.val_dir, self.test_dir, batch_size=batch_size
            )
            
            # Initialize model
            model = get_model(dropout_rate=dropout_rate, device=device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            
            # Learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            
            # Training loop
            best_val_accuracy = 0.0
            
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                # Training phase
                for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
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
                
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
                
                # Update best validation accuracy
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
            
            return best_val_accuracy
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return 0.0  # Return 0 accuracy on error
    
    def optimize(self, epochs_per_iteration=5):
        """
        Run the PSO optimization algorithm
        Args:
            epochs_per_iteration (int): Number of epochs to train each model
        Returns:
            best_hyperparams (dict): Best hyperparameters found
            best_fitness (float): Best fitness value (validation accuracy)
        """
        start_time = time.time()
        
        # Initialize progress bar
        pbar = tqdm(total=self.max_iterations, desc="PSO Optimization")
        
        for iteration in range(self.max_iterations):
            for i, particle in enumerate(self.particles):
                # Evaluate current particle
                print(f"\nParticle {i+1}/{self.num_particles}, Iteration {iteration+1}/{self.max_iterations}")
                fitness = self.evaluate_particle(particle, epochs=epochs_per_iteration)
                particle.current_fitness = fitness
                
                # Update particle's best position if current position is better
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                    
                    # Update global best if this particle's best is better
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = particle.position.copy()
                        
                        # Convert to hyperparameters for display
                        best_hyperparams = {}
                        for j, key in enumerate(self.param_keys):
                            if key == 'batch_size':
                                best_hyperparams[key] = int(self.global_best_position[j])
                            else:
                                best_hyperparams[key] = self.global_best_position[j]
                        
                        print(f"\n=== New Global Best ===")
                        print(f"Accuracy: {self.global_best_fitness:.2f}%")
                        print(f"Hyperparameters: {best_hyperparams}")
            
            # Update velocities and positions for all particles
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position(self.bounds)
            
            # Update progress bar
            pbar.update(1)
            
            # Save intermediate results
            if (iteration + 1) % 5 == 0 or iteration == self.max_iterations - 1:
                self.save_results(iteration + 1)
        
        pbar.close()
        
        # Convert best position to hyperparameters
        best_hyperparams = {}
        for i, key in enumerate(self.param_keys):
            if key == 'batch_size':
                best_hyperparams[key] = int(self.global_best_position[i])
            else:
                best_hyperparams[key] = self.global_best_position[i]
        
        # Print final results
        print("\n=== PSO Optimization Complete ===")
        print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.global_best_fitness:.2f}%")
        print(f"Best hyperparameters: {best_hyperparams}")
        
        return best_hyperparams, self.global_best_fitness
    
    def save_results(self, iteration):
        """
        Save intermediate results to a file
        Args:
            iteration (int): Current iteration number
        """
        # Convert best position to hyperparameters
        best_hyperparams = {}
        for i, key in enumerate(self.param_keys):
            if key == 'batch_size':
                best_hyperparams[key] = int(self.global_best_position[i])
            else:
                best_hyperparams[key] = self.global_best_position[i]
        
        # Save to file
        with open('pso_results.txt', 'a') as f:
            f.write(f"Iteration {iteration}:\n")
            f.write(f"Best accuracy: {self.global_best_fitness:.2f}%\n")
            f.write(f"Best hyperparameters: {best_hyperparams}\n\n")
