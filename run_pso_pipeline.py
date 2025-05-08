import os
import argparse
import subprocess
import time
import sys

def parse_args():
    """
    Parse command line arguments
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run PSO Optimization Pipeline')
    
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
    
    parser.add_argument('--original_model', type=str, default='plant_disease_model.pt',
                        help='Path to original model weights')
    parser.add_argument('--optimized_model', type=str, default='pso_optimized_model.pt',
                        help='Path to save the PSO-optimized model')
    
    parser.add_argument('--skip_optimization', action='store_true',
                        help='Skip the optimization step (use if already optimized)')
    parser.add_argument('--skip_comparison', action='store_true',
                        help='Skip the comparison step')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip the visualization step')
    
    return parser.parse_args()

def run_command(command, description):
    """
    Run a command and print its output
    Args:
        command (list): Command to run
        description (str): Description of the command
    Returns:
        success (bool): Whether the command succeeded
    """
    print(f"\n{'='*20} {description} {'='*20}\n")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            sys.stdout.flush()
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n{description} completed successfully!")
            return True
        else:
            print(f"\n{description} failed with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"\n{description} failed with error: {str(e)}")
        return False

def check_files_exist(files):
    """
    Check if files exist
    Args:
        files (list): List of file paths to check
    Returns:
        all_exist (bool): Whether all files exist
    """
    all_exist = True
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """
    Main function to run the PSO optimization pipeline
    """
    # Parse arguments
    args = parse_args()
    
    # Check if data directories exist
    if not check_files_exist([args.train_dir, args.val_dir, args.test_dir]):
        print("Data directories not found. Please check the paths.")
        return
    
    # Check if original model exists
    if not os.path.exists(args.original_model):
        print(f"Original model not found: {args.original_model}")
        print("Please train the original model first.")
        return
    
    start_time = time.time()
    
    # Step 1: Run PSO Optimization
    if not args.skip_optimization:
        optimize_command = [
            'python', 'optimize_model.py',
            '--train_dir', args.train_dir,
            '--val_dir', args.val_dir,
            '--test_dir', args.test_dir,
            '--num_particles', str(args.num_particles),
            '--max_iterations', str(args.max_iterations),
            '--epochs_per_iteration', str(args.epochs_per_iteration),
            '--final_epochs', str(args.final_epochs),
            '--model_save_path', args.optimized_model
        ]
        
        if not run_command(optimize_command, "PSO Optimization"):
            print("PSO Optimization failed. Stopping pipeline.")
            return
    else:
        print("\nSkipping PSO Optimization step...")
        
        # Check if optimized model exists
        if not os.path.exists(args.optimized_model):
            print(f"Optimized model not found: {args.optimized_model}")
            print("Please run the optimization step or provide the correct path.")
            return
    
    # Step 2: Visualize PSO Results
    if not args.skip_visualization:
        visualize_command = [
            'python', 'visualize_pso.py',
            '--results_file', 'pso_results.txt',
            '--hyperparams_file', 'best_hyperparams.json'
        ]
        
        if not run_command(visualize_command, "PSO Visualization"):
            print("PSO Visualization failed. Continuing with comparison...")
    else:
        print("\nSkipping PSO Visualization step...")
    
    # Step 3: Compare Models
    if not args.skip_comparison:
        compare_command = [
            'python', 'compare_models.py',
            '--original_model', args.original_model,
            '--optimized_model', args.optimized_model,
            '--train_dir', args.train_dir,
            '--val_dir', args.val_dir,
            '--test_dir', args.test_dir
        ]
        
        if not run_command(compare_command, "Model Comparison"):
            print("Model Comparison failed.")
    else:
        print("\nSkipping Model Comparison step...")
    
    # Calculate total time
    total_time = (time.time() - start_time) / 60
    
    print("\n" + "="*50)
    print(f"PSO Pipeline completed in {total_time:.2f} minutes")
    print("="*50)
    
    # Print summary of output files
    print("\nOutput Files:")
    output_files = [
        'pso_results.txt',
        'best_hyperparams.json',
        args.optimized_model,
        'training_history.png',
        'accuracy_progress.png',
        'hyperparameter_evolution.png',
        'hyperparameter_correlation.png',
        'hyperparameter_space_3d.png',
        'confusion_matrix_original.png',
        'confusion_matrix_optimized.png',
        'confidence_distribution.png',
        'accuracy_comparison.png',
        'model_comparison_results.txt'
    ]
    
    for file in output_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (not found)")
    
    print("\nNext Steps:")
    print("1. Review the model comparison results in 'model_comparison_results.txt'")
    print("2. Check the visualization plots to understand the PSO optimization process")
    print("3. Use the optimized model in your GUI application by updating the model path")
    print("4. Consider further fine-tuning or ensemble methods for even better results")

if __name__ == "__main__":
    main()
