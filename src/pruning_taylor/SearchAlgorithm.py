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

def new_algo(original_mode, min_accuracy):
    
    '''
    loss function would
    where
    accuracy with max importance
    next computation time
    then model size
    
    also if org acc is 50% and we need to bring it down to 20%
    then we compute the ratio between then 50/20 * 100 = 25
    we start with a pruning ratio of 25% rather than start at the mid
    25% may not be the best option but it converges faster
    
    
    between 0 and 1
    acc = 0.7
    computation_time = 0.25
    model_size = 0.05
    
    reward function for accuracy:
    if acc > threshold: reward -0.2 * (acc - threshold)
    if acc ≈ threshold: reward 10
    if acc < threshold: reward -1000
    
    reward function fot computation time:
    if comp_time < threshold: reward 10
    else reward: -100
    
    reward function for model size:
    if model_size < org model size: reward 10
    else reward: -10
    
    
    
    
    but then we have lets say a set of best_percentages
    lets say [20.0, 20.3, 20.5, 21.04]
    we can then take the one with the lowest loss function
    we can store the last 20 accuracies, comp times and model size in a list
    [pruning amount, acc, computation time, model size] or we can have the reward function instead
        
    
    
    
    + 0.25 * comp_time + 0.05 * model_size
    
    '''
    return