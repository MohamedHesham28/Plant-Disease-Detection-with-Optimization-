# PSO Optimization for Plant Disease Detection Model

This project implements Particle Swarm Optimization (PSO) to improve the accuracy and confidence of the plant disease detection model without overfitting.

## What is PSO?

Particle Swarm Optimization (PSO) is a population-based stochastic optimization technique inspired by the social behavior of birds flocking or fish schooling. In PSO, each particle represents a potential solution to the optimization problem, and the swarm of particles moves through the solution space to find the optimal solution.

## Benefits of PSO for Neural Network Optimization

1. **Global Search**: PSO can effectively search the entire solution space to find the global optimum, avoiding local minima.
2. **No Gradient Information Required**: Unlike gradient-based methods, PSO doesn't require gradient information, making it suitable for non-differentiable or discontinuous problems.
3. **Simplicity**: PSO is easy to implement and has few parameters to tune.
4. **Parallelizability**: PSO can be easily parallelized, making it efficient for large-scale optimization problems.

## Files in this Project

- `pso_optimizer.py`: Implementation of the PSO algorithm for hyperparameter optimization
- `optimize_model.py`: Script to run the PSO optimization and train the final model
- `compare_models.py`: Script to compare the original and PSO-optimized models
- `PSO_README.md`: This README file

## Hyperparameters Optimized by PSO

The PSO algorithm optimizes the following hyperparameters:

1. **Learning Rate**: Controls the step size during optimization (range: 0.0001 to 0.01)
2. **Batch Size**: Number of samples processed before the model is updated (range: 16 to 64)
3. **Dropout Rate**: Probability of dropping neurons during training to prevent overfitting (range: 0.1 to 0.5)
4. **Weight Decay**: L2 regularization parameter to prevent overfitting (range: 1e-6 to 1e-3)

## How to Use

### Step 1: Run PSO Optimization

```bash
python optimize_model.py --train_dir "plant disease dataset/train" --val_dir "plant disease dataset/validation" --test_dir "plant disease dataset/test" --num_particles 10 --max_iterations 10 --epochs_per_iteration 5 --final_epochs 20 --model_save_path "pso_optimized_model.pt"
```

Parameters:
- `--train_dir`: Path to training data directory
- `--val_dir`: Path to validation data directory
- `--test_dir`: Path to test data directory
- `--num_particles`: Number of particles in the swarm (default: 10)
- `--max_iterations`: Maximum number of PSO iterations (default: 10)
- `--epochs_per_iteration`: Number of epochs to train each model during PSO (default: 5)
- `--final_epochs`: Number of epochs to train the final model (default: 20)
- `--model_save_path`: Path to save the optimized model (default: "pso_optimized_model.pt")

### Step 2: Compare Models

After running the PSO optimization, you can compare the original and PSO-optimized models:

```bash
python compare_models.py --original_model "plant_disease_model.pt" --optimized_model "pso_optimized_model.pt" --train_dir "plant disease dataset/train" --val_dir "plant disease dataset/validation" --test_dir "plant disease dataset/test"
```

Parameters:
- `--original_model`: Path to original model weights (default: "plant_disease_model.pt")
- `--optimized_model`: Path to PSO-optimized model weights (default: "pso_optimized_model.pt")
- `--train_dir`: Path to training data directory
- `--val_dir`: Path to validation data directory
- `--test_dir`: Path to test data directory
- `--batch_size`: Batch size for evaluation (default: 32)

### Step 3: Use the Optimized Model in the GUI

The PSO-optimized model can be used in the GUI application by modifying the model path in the GUI code:

```python
# Load the trained model
model_path = 'pso_optimized_model.pt'  # Change this to the PSO-optimized model path
```

## Output Files

The PSO optimization process generates the following output files:

1. `pso_results.txt`: Intermediate results of the PSO optimization process
2. `best_hyperparams.json`: Best hyperparameters found by PSO
3. `pso_optimized_model.pt`: Trained model with the best hyperparameters
4. `training_history.png`: Plot of training and validation accuracy/loss
5. `confusion_matrix_original.png`: Confusion matrix for the original model
6. `confusion_matrix_optimized.png`: Confusion matrix for the PSO-optimized model
7. `confidence_distribution.png`: Comparison of prediction confidence distributions
8. `accuracy_comparison.png`: Comparison of model accuracies
9. `model_comparison_results.txt`: Summary of model comparison results

## Expected Improvements

The PSO-optimized model is expected to show improvements in:

1. **Accuracy**: Higher classification accuracy on the test set
2. **Confidence**: Higher prediction confidence
3. **Generalization**: Better performance on unseen data
4. **Robustness**: More consistent predictions across different classes

## Tips for Better Results

1. **Increase Particles and Iterations**: More particles and iterations can lead to better results but will take longer to run
2. **Adjust Hyperparameter Ranges**: If the optimal values are consistently at the boundaries, consider expanding the search ranges
3. **Use Early Stopping**: To prevent overfitting during the final model training
4. **Ensemble Models**: Consider creating an ensemble of the best models found during PSO

## Troubleshooting

1. **Out of Memory Errors**: Reduce batch size or model complexity
2. **Slow Convergence**: Increase the number of particles or iterations
3. **Overfitting**: Increase dropout rate or weight decay ranges
4. **Underfitting**: Decrease dropout rate or increase learning rate ranges

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. Proceedings of ICNN'95 - International Conference on Neural Networks.
2. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer. IEEE International Conference on Evolutionary Computation.
3. Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation.
