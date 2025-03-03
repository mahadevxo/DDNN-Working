import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from PruningFineTuner import PruningFineTuner

def get_candidates_to_prune_order(fine_tuner, num_filters, order):
    # Override ranking based on Taylor order
    fine_tuner.pruner.reset()
    fine_tuner.train_epoch(rank_filter=True)
    fine_tuner.pruner.normalize_ranks_per_layer()
    if order == 2:
        fine_tuner.pruner.filter_ranks = fine_tuner.pruner.filter_ranks_2nd
    elif order == 3:
        fine_tuner.pruner.filter_ranks = fine_tuner.pruner.filter_ranks_3rd
    return fine_tuner.pruner.get_pruning_plan(num_filters)

def run_pruning_experiment(taylor_order, pruning_ratios):
    accuracies = []
    for ratio in pruning_ratios:
        # Load a fresh VGG16 model
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        tuner = PruningFineTuner(model)
        # Monkey-patch to use the desired ranking (only for Taylor order 2 and 3)
        if taylor_order in [2, 3]:
            tuner.get_candidates_to_prune = lambda num: get_candidates_to_prune_order(tuner, num, taylor_order)
        # Prune and fine tune; prune() returns [acc_pre, acc_post, compute_time, model_size]
        result = tuner.prune(ratio)
        final_top1 = result[1]
        print(f"Taylor order {taylor_order} - Pruning {ratio}%: Top-1 Accuracy = {final_top1:.2f}%")
        accuracies.append(final_top1)
    return accuracies

def main():
    pruning_ratios = [0, 10, 15, 20, 30, 40, 50, 60]
    orders = [1, 2, 3]
    results = {
        order: run_pruning_experiment(order, pruning_ratios)
        for order in orders
    }
    # Plot accuracy vs pruning ratio for each Taylor order
    for order in orders:
        plt.plot(pruning_ratios, results[order], marker='o', label=f'Taylor Order {order}')
    plt.xlabel('Pruning Ratio (%)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Accuracy vs Pruning Ratio for VGG16')
    plt.legend()
    plt.grid(True)
    plt.imsave('pruning_taylor_higher_order.png')

if __name__ == '__main__':
    main()
