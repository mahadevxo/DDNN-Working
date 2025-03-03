import torchvision.models as models
import matplotlib.pyplot as plt
from PruningFineTuner import PruningFineTuner

def main():
    # Instantiate a pretrained model for Taylor Order 1 pruning
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    pruning_ratios = [0, 10, 20, 30, 40, 50, 60]
    top1_accuracies = []

    with open ('results_1.csv', 'w') as f:
        f.write('Pruning Ratio, Top-1 Accuracy\n')
        for ratio in pruning_ratios:
            tuner = PruningFineTuner(model)
            results = tuner.prune(ratio)  # results: [pre_fine_tuning_acc, raw_acc, compute_time, size_mb]
            top1_accuracies.append(results[1])
            data = f"Pruning {ratio}%: Top-1 Accuracy = {results[1]*100:.2f}%"
            print(data)
            f.write(f"{ratio}, {results[1]}\n")
            # Reload the model for the next iteration
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
if __name__ == '__main__':
    main()
