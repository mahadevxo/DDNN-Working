import torchvision.models as models
from PruningFineTuner import PruningFineTuner
import numpy as np

def main():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # try:
    #     pruning_ratios = [int(input("Enter the pruning ratio: "))]
    #     print(pruning_ratios)
    # except Exception as e:
    #     pruning_ratios = None
    #     print(f"Error: {str(e)}")
    #     print("No pruning ratio entered. Using default values.")
    # finally:
    #     if pruning_ratios is None:
    #         pruning_ratios = np.arange(0, 100, 10)
    
    pruning_ratios = np.arange(0, 100, 10)

    with open('results_3.csv', 'w') as f:
        f.write('Pruning Ratio, Accuracy, Compute Time, Model Size\n')
        for ratio in pruning_ratios:
            try:
                tuner = PruningFineTuner(model, 3) # Taylor Order 3
                def get_candidates_to_prune_3nd(num_filter_to_prune):
                    tuner.pruner.reset()
                    tuner.train_epoch(rank_filter=True)
                    tuner.pruner.filter_ranks = tuner.pruner.filter_ranks_3rd
                    tuner.pruner.normalize_ranks_per_layer()
                    return tuner.pruner.get_pruning_plan(num_filter_to_prune)
                tuner.get_candidates_to_prune = get_candidates_to_prune_3nd

                results = tuner.prune(ratio) 
                data = f"{ratio, results[1], results[2], results[3]}"
                print(data)
                f.write(f"{data}\n")
            except Exception as e:
                f.write(f"{ratio}, -1, -1, -1\n")
                print(f"Error: {str(e)}")
                continue
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

if __name__ == '__main__':
    main()
