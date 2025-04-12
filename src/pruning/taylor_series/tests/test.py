import torch
from torchvision import models as models
import numpy as np
import sys
import gc
import time

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def create_results_file(filename):
    with open(filename, 'w') as f:
        f.write("Pruning Amount, Final Accuracy, Time, Memory\n")
    return filename

def append_result(filename, pruning_amount, accuracy, compute_time, model_size):
    with open(filename, 'a') as f:
        f.write(f"{pruning_amount}%, {accuracy}%,{compute_time}, {model_size}\n")

def main():
    pruning_amounts = np.arange(0, 60, 0.5)
    
    # Determine device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create results file
    
    models_selected = int(input("1: VGG13\n2: VGG16\n3: VGG19\n4: AlexNet\nEnter Option: "))
    model_save = bool(int(input("0: Test Model\n1: Save Model\nEnter Option: ")))
    
    if not model_save:
        results_filename = create_results_file()
    
    try:
        for pruning_amount in pruning_amounts:
            clear_memory()
            
            if 'PruningFineTuner' in sys.modules:
                del sys.modules['PruningFineTuner']
            from PruningFineTuner import PruningFineTuner
            
            # Load a fresh model for each pruning amount
            print(f"\n{'='*50}\nStarting pruning at {pruning_amount:.2f}%\n{'='*50}")
            model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1).to(device) if models_selected == 1 else \
                   models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device) if models_selected == 2 else \
                   models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device) if models_selected == 3 else \
                   models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device) if models_selected == 4 else \
                   models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
            
            model_name = "VGG13" if models_selected == 1 else \
                            "VGG16" if models_selected == 2 else \
                            "VGG19" if models_selected == 3 else \
                            "AlexNet" if models_selected == 4 else \
                            "VGG16"
            
            pruning_fine_tuner = PruningFineTuner(model)
            
            pruning_fine_tuner.prune(pruning_percentage=pruning_amount)
            
            if model_save:
                pruning_fine_tuner.save_model(f'pruned_{model_name}_{pruning_amount:.2f}.pth')
            else:
                out = pruning_fine_tuner.test_model(pruning_fine_tuner.model, final_test = True)
                model_size = pruning_fine_tuner.get_model_size(model)
                
                append_result(results_filename, pruning_amount, out[0], out[1], model_size)
                print(f"Pruning Amount: {pruning_amount:.2f}%, Final Accuracy: {out[0]:.3f}%, Time: {out[1]:.5f}, Memory: {model_size:.3f}")
            
            
            pruning_fine_tuner.reset()
            del pruning_fine_tuner
            del model
            
            clear_memory()
            
    except Exception as e:
        print(f"Error during pruning: {str(e)}")
        import traceback
        traceback.print_exc()
    if not model_save:
        print(f"Model saved as pruned_{model_name}_{pruning_amount:.2f}.pth")
    print(f"Results saved to {results_filename}")
        
if __name__ == '__main__':
    main()