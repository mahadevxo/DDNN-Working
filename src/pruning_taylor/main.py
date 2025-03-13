import torchvision.models as models
from SearchAlgorithm import SearchAlgorithm
from PruningFineTuner import PruningFineTuner
import torch
def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    pruning_fine_tuner = PruningFineTuner(model)
    out = pruning_fine_tuner.test(model)
    print(f"Original Accuracy: {out[0]:.2f}%")
    print(f'Original Model Size: {pruning_fine_tuner.get_model_size(model):.2f} MB')
    del pruning_fine_tuner
    min_acc = float(input("Enter minimum acceptable accuracy (0-100): "))
    print(f"Minimum Accuracy: {min_acc}%")
    searching_strategy = SearchAlgorithm(model, min_accuracy=(min_acc/100))
    opt = int(input("Enter optimizer (0: Adam, 1: Binary): "))
    if opt == 0:
        best_percentage = searching_strategy.heuristic_adam()
    else:
        best_percentage = searching_strategy.heuristic_binary_search()    
    
    print(f"Recommended pruning percentage: {best_percentage:.2f}%")

if __name__ == '__main__':
    main()