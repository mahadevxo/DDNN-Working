import torch
from torchvision.models import models
from PruningFineTuner import PruningFineTuner
def heuristic_search(original_model, min_accuracy, max_iter=6):
    """
    Heuristic search to find the best pruning percentage that gets final accuracy
    as close to (but not lower than) min_accuracy, optimized for lower memory usage.
    """
    original_state = original_model.state_dict()
    lower, upper = 0.0, 100.0
    best_percentage = 0.0
    best_final_acc = 0.0
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        trial_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        trial_model.load_state_dict(original_state)
        pruner = PruningFineTuner(trial_model)
        pruner.prune(pruning_percentage=mid)
        final_acc = pruner.test(pruner.model)[0]
        if final_acc >= min_accuracy:
            best_percentage = mid
            best_final_acc = final_acc
            lower = mid
        else:
            upper = mid
        del pruner
        del trial_model
        if device == 'cuda':
            torch.cuda.empty_cache()
    print(f"Best pruning percentage: {best_percentage:.2f}% yields final accuracy: {best_final_acc*100:.2f}%")
    return best_percentage