import torchvision.models as models
from PruningFineTuner import PruningFineTuner

def main():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    pruning_ratios = [0, 10, 20, 30, 40, 50, 60]

    with open('results_2.csv', 'w') as f:
        f.write('Pruning Ratio, Accuracy, Model Size, Compute Time\n')
        for ratio in pruning_ratios:
            tuner = PruningFineTuner(model, 2) # Taylor Order 2
            def get_candidates_to_prune_2nd(num_filter_to_prune):
                tuner.pruner.reset()
                tuner.train_epoch(rank_filter=True)
                tuner.pruner.filter_ranks = tuner.pruner.filter_ranks_2nd
                tuner.pruner.normalize_ranks_per_layer()
                return tuner.pruner.get_pruning_plan(num_filter_to_prune)
            tuner.get_candidates_to_prune = get_candidates_to_prune_2nd

            results = tuner.prune(ratio) 
            # results: [pre_fine_tuning_acc, acc 1%, 5%, compute_time, model_size]
            data = f"{results[1]}, {results[3]}, {results[2]}"
            print(data)
            f.write(f"{ratio}, {data}\n")
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

if __name__ == '__main__':
    main()
