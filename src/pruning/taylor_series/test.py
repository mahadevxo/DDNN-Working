import torch
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import sys
import os
import gc
import time

def clear_memory():
    """Helper function to clear memory between pruning runs"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def create_results_file(filename='pruning_results.csv'):
    """Create results file with headers"""
    with open(filename, 'w') as f:
        f.write("Pruning Amount, Final Accuracy, Time, Memory\n")
    return filename

def append_result(filename, pruning_amount, accuracy, compute_time, model_size):
    """Safely append a result to the CSV file"""
    with open(filename, 'a') as f:
        f.write(f"{pruning_amount}%, {accuracy}%,{compute_time}, {model_size}\n")

def main():
    pruning_amounts = np.arange(0, 60, 0.5)
    # pruning_amounts = np.arange(0, 60, 10)
    
    # Determine device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create results file
    results_filename = create_results_file()
    
    try:
        for pruning_amount in pruning_amounts:
            # Force Python garbage collection
            clear_memory()
            
            # Import inside loop to ensure clean module state
            if 'PruningFineTuner' in sys.modules:
                del sys.modules['PruningFineTuner']
            from PruningFineTuner import PruningFineTuner
            
            # Load a fresh model for each pruning amount
            print(f"\n{'='*50}\nStarting pruning at {pruning_amount:.2f}%\n{'='*50}")
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
            
            # Initialize pruner
            pruning_fine_tuner = PruningFineTuner(model)
            
            # Run pruning
            pruning_fine_tuner.prune(pruning_percentage=pruning_amount)
            
            # Get performance metrics
            out = pruning_fine_tuner.test(pruning_fine_tuner.model)
            model_size = pruning_fine_tuner.get_model_size(model)
            
            # Save results (even if we crash later)
            append_result(results_filename, pruning_amount, out[0], out[1], model_size)
            print(f"Pruning Amount: {pruning_amount:.2f}%, Final Accuracy: {out[0]:.2f}%, Time: {out[1]:.2f}, Memory: {model_size:.2f}")
            
            # Cleanup thoroughly
            pruning_fine_tuner.reset()
            del pruning_fine_tuner
            del model
            
            clear_memory()
            
    except Exception as e:
        print(f"Error during pruning: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"Results saved to {results_filename}")
        
if __name__ == '__main__':
    main()