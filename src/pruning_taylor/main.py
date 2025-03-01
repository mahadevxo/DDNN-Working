import torchvision.models as models
from PruningFineTuner import PruningFineTuner
from SearchAlgorithm import heuristic_search
import torch
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    pruning_fine_tuner = PruningFineTuner(model)
    print(f"Original Accuracy: {(pruning_fine_tuner.test(model)[0]*100):.2f}%")
    print(f'Original Model Size: {pruning_fine_tuner.get_model_size(model):.2f} MB')
    del pruning_fine_tuner
    min_acc = float(input("Enter minimum acceptable accuracy (0-1): "))
    print(f"Minimum Accuracy: {min_acc*100}%")
    best_percentage = heuristic_search(model, min_acc)
    print(f"Recommended pruning percentage: {best_percentage:.2f}%")

if __name__ == '__main__':
    main()