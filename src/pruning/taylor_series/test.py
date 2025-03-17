import torch
from torchvision import models
from PruningFineTuner import PruningFineTuner
import numpy as np

def main():
    pruning_amounts = np.arange(0, 60, 0.25)
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    text = "Pruning Amount, Final Accuracy, Time, Memory\n"
    for pruning_amount in pruning_amounts:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        pruning_fine_tuner = PruningFineTuner(model)
        pruning_fine_tuner.prune(pruning_percentage=pruning_amount)
        out = pruning_fine_tuner.test(model)
        model_size = pruning_fine_tuner.get_model_size(model)
        text += f"{pruning_amount:.2f}%, {out[0]}%,{out[1]:.2f}, {model_size}\n"
        print(f"Pruning Amount: {pruning_amount:.2f}%, Final Accuracy: {out[0]:.2f}%, Time: {out[1]:.2f}, Memory: {model_size:.2f}")
        del pruning_fine_tuner
        model.cpu()
        del model
        torch.cuda.empty_cache()
    
    with open('pruning_results.csv', 'w') as f:
        f.write(text)
    print("Results saved to pruning_results.csv")
        
if __name__ == '__main__':
    main()