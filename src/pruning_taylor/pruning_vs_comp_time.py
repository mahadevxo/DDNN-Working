from torchvision import models as models
from PruningFineTuner import PruningFineTuner
import numpy
import torch
import gc
import time

def free_memory():
    """Aggressively free memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    pruning_amounts = numpy.arange(0.0, 60.0, 2.0)
    print(f"Number of pruning percentages: {len(pruning_amounts)}")
    
    with open("pruning_vs_comp_time.csv", mode='w') as file:
        file.write("Pruning Amount, Pre-Accuracy, Accuracy, Pre-Pruning Time, Post-Pruning Time, Size\n")
        for i, pruning_amount in enumerate(pruning_amounts):
            try:
                print(f"Starting iteration {i+1}/{len(pruning_amounts)} for pruning amount {pruning_amount}%")
                
                # Clear memory before creating a new model
                free_memory()
                
                with torch.no_grad():  # Use no_grad for initial model setup
                    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
                    model.eval()  # Set to eval mode to prevent storing stats
                
                pruning_fine_tuner = PruningFineTuner(model)
                
                try:
                    data = pruning_fine_tuner.prune(pruning_amount)
                    pre = data[0]
                    pre_acc, pre_comp = pre[0], pre[1]
                    acc, comp_time, size = data[1], data[2], data[3]
                    entry = f"{pruning_amount}, {pre_acc}, {acc}, {pre_comp}, {comp_time}, {size}"
                    print(entry)
                    file.write(entry + '\n')
                    file.flush()
                except Exception as e:
                    print(f"Error during pruning at {pruning_amount}%: {str(e)}")
                    file.write(f"{pruning_amount}, ERROR, ERROR, ERROR, ERROR, ERROR\n")
                    file.flush()
                
                # Proper cleanup after each iteration
                pruning_fine_tuner.reset()
                
                if 'data' in locals():
                    del data
                if 'pre' in locals():
                    del pre
                
                del pruning_fine_tuner
                del model
                free_memory()
                
                # Add a small delay to ensure memory is properly released
                time.sleep(1)
                
                if i % 3 == 0 and device == 'cuda':
                    print(f"CUDA Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"CUDA Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            except Exception as e:
                print(f"Fatal error in iteration {i+1}: {str(e)}")
                continue  # Try to continue with the next iteration
    
    print("Done")

if __name__ == '__main__':
    main()