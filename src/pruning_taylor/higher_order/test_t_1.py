import torchvision.models as models
from PruningFineTuner import PruningFineTuner
import numpy as np
def main():
    # Instantiate a pretrained model for Taylor Order 1 pruning
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    pruning_ratios = np.arange(0, 100, 10)

    with open ('results_1.csv', 'w') as f:
        f.write('Pruning Ratio, Accuracy, Compute Time, Model Size\n')
        for ratio in pruning_ratios:
            try:
                tuner = PruningFineTuner(model, 1) # Taylor Order x1
                results = tuner.prune(ratio)  # results: [pre_fine_tuning_acc, raw_acc, compute_time, size_mb]
                data = f"{ratio, results[1], results[2], results[3]}"
                print(data)
                f.write(f"{data}\n")
                # Reload the model for the next iteration
            except Exception as e:
                f.write(f"{ratio}, -1, -1, -1\n")
                print(f"Error: {str(e)}")
                continue
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        f.close()
        
if __name__ == '__main__':
    main()
