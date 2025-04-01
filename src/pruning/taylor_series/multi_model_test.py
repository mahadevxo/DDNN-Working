import torch
import torchvision.models as models
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
        f.write("Pruning Amount, Pre Fine Tuning Accuracy, Inference Time Pre Pruning, Final Accuracy, Compute Time, Model Size\n")
    print(f"Results file created: {filename}")
    return filename

def append_result(filename, pruning_amount, acc_pre_fine_tuning, inference_time_pre_pruning, final_accuracy, compute_time, model_size):
    with open(filename, 'a') as f:
        f.write(f"{pruning_amount}, {acc_pre_fine_tuning:.2f}, {inference_time_pre_pruning:.2f}, {final_accuracy:.2f}, {compute_time:.2f}, {model_size:.2f}\n")

def get_model(model_name):
    if model_name == "VGG11":
        return models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    elif model_name == "VGG13":
        return models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    elif model_name == "VGG16":
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == "VGG19":
        return models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == "AlexNet":
        return models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def main():
    pruning_amounts = np.arange(0, 60, 1)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models = [
        'VGG11',
        'VGG13',
        'VGG16',
        'VGG19',
        'AlexNet',
    ]
    for model_name in models:
        print(f"Using model: {model_name}")
        create_results_file(f"{model_name}_results.csv")
        try:
            for pruning_amount in pruning_amounts:
                clear_memory()
                # Load a fresh model for each pruning amount
                model = get_model(model_name).to(device)
                if 'PruningFineTuner' in sys.modules:
                    del sys.modules['PruningFineTuner']
                from PruningFineTuner import PruningFineTuner
                
                print(f"\n{'='*50}\nStarting pruning at {pruning_amount:.2f}%\n{'='*50}")
                
                # Initialize the PruningFineTuner
                pruning_fine_tuner = PruningFineTuner(model)
                
                prune_out = pruning_fine_tuner.prune(pruning_percentage=pruning_amount)
                
                acc_pre_fine_tuning = prune_out[0][0]
                inference_time_pre_pruning = prune_out[0][1]
                
                final_accuracy, compute_time = pruning_fine_tuner.test(pruning_fine_tuner.model, final_test=True)
                model_size = pruning_fine_tuner.get_model_size(model)
                
                print(f"Pruning amount: {pruning_amount:.2f}%, Pre Fine Tuning Accuracy: {acc_pre_fine_tuning:.2f}%, Inference Time Pre Pruning: {inference_time_pre_pruning:.2f}, Final Accuracy: {final_accuracy:.2f}%, Compute Time: {compute_time:.2f}, Model Size: {model_size:.2f}")
                
                append_result(f"{model_name}_results.csv", pruning_amount, acc_pre_fine_tuning, inference_time_pre_pruning, final_accuracy, compute_time, model_size)
                
                print(f"Results appended to {model_name}_final_results.csv")
                
                pruning_fine_tuner.reset()
                del pruning_fine_tuner
                del model
                clear_memory()
                
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            continue
        
if __name__ == '__main__':
    main()