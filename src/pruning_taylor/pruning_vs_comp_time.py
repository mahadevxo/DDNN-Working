from torchvision import models as models
from PruningFineTuner import PruningFineTuner
import numpy
import torch

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    pruning_fine_tuner = PruningFineTuner(model)
    
    pruning_amounts = numpy.arange(0.0, 100.0, 5.0)
    print(f"Number of pruning percentages: {len(pruning_amounts)}")
    
    with open("pruning_vs_comp_time.csv", mode='w') as file:
        file.write("Pruning Percentage, Accuracy Before Fine Tuning, Accuracy, Compute Time, Model Size\n")
        for pruning_amount in pruning_amounts:
            data = pruning_fine_tuner.prune(pruning_amount)
            print(data)
            data = ', '.join(map(str, data))
            file.write(data + '\n')
    
    print("Done")


if __name__ == '__main__':
    main()