from torchvision import models as models
from PruningFineTuner import PruningFineTuner
import numpy
import torch

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    pruning_fine_tuner = PruningFineTuner(model)
    
    pruning_amounts = numpy.linspace(0.1, 0.9, 9)
    
    with open("pruning_vs_comp_time.csv", mode='w') as file:
        for pruning_amount in pruning_amounts:
            acc, time, model_size = pruning_fine_tuner.prune(pruning_amount)
            data = (f"Pruning Percentage: {pruning_amount:.2f}%, Accuracy: {acc:.2f}, Compute Time: {time:.2f}, Model Size: {model_size:.2f} MB")
            print(data)
            file.write(data + '\n')
    
    print("Done")


if __name__ == '__main__':
    main()