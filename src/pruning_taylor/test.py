import torch
from torchvision import models
from PruningFineTuner import PruningFineTuner

def main():
    pruning_amount = 5
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    pruning_fine_tuner = PruningFineTuner(model)
    pruning_fine_tuner.prune(pruning_percentage=pruning_amount)
    out = pruning_fine_tuner.test(model)
    print(f"Original Accuracy: {out[0]*100:.2f}%")
    print(f'Original Model Size: {pruning_fine_tuner.get_model_size(model):.2f} MB')
    
if __name__ == '__main__':
    main()