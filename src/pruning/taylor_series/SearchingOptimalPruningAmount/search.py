import gc
from models import MVCNN
import torch
from prune import get_ranks, get_pruned_model
from train import validate_model, fine_tune
from Rewards import Reward
import cma
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

device: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
comp_time_last: float = None

def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def _get_num_filters(model: torch.nn.Module) -> int:
    return sum(
        layer.out_channels
        for layer in model.net_1
        if isinstance(layer, torch.nn.Conv2d)
    )

def _get_model_size(model):
    """Calculate model size more accurately including only required parameters"""
    from train import get_model_size_by_params
    return get_model_size_by_params(model)

def get_model() -> torch.nn.Module:
    global device
    model: torch.nn.Module = MVCNN.SVCNN('SVCNN')
    weights: dict = torch.load('./model-00030.pth', map_location=device)
    model.load_state_dict(weights)
    model: torch.nn.Module = model.to(device)
    del weights
    _clear_memory()
    return model

def get_Reward(pruning_amount: float, ranks: tuple, rewardfn: Reward) -> tuple:
    global device
    base_model = get_model()                                                     
    original_size = _get_model_size(base_model)
    
    pruned_model = get_pruned_model(ranks=ranks, model=base_model, pruning_amount=pruning_amount)
    pruned_model = pruned_model.to(device)
    pruned_size = _get_model_size(pruned_model)
    
    print(f"Filters of Pruned Model: {_get_num_filters(pruned_model)}")
    print(f"Original size: {original_size:.2f}MB, Pruned size: {pruned_size:.2f}MB, Reduction: {(1 - pruned_size/original_size)*100:.1f}%")
    
    finetuned_model, _ = fine_tune(pruned_model, rank_filter=False)
    accuracy, time, model_size = validate_model(finetuned_model)
    
    print(f"Filters of Fine-tuned Model: {_get_num_filters(finetuned_model)}")
    print(f"Filters of Base Model: {_get_num_filters(base_model)}")
    
    print('-'*20)
    
    print(f"Accuracy: {accuracy:.2f}%, Time: {time:.2f}s, Model Size: {model_size:.4f}MB")
    
    # Force cleanup of models
    del base_model
    del pruned_model
    del finetuned_model
    _clear_memory()
    
    global comp_time_last
    if comp_time_last is None:
        comp_time_last = time
    
    reward, comp_time = rewardfn.getReward(accuracy=accuracy, comp_time=time, model_size=model_size, comp_time_last=comp_time_last)
    comp_time_last = comp_time
    _clear_memory()
    
    print('-'*20)
    return reward

def main() -> None:
    # res = validate_model(get_model())
    # print(f"Initial Validation Accuracy: {res[0]:.2f}%, Time: {res[1]:.6f}s, Model Size: {res[2]:.2f}MB")
    min_acc: float = 50
    min_size: float = 300
    x: float = 0.7  # Higher weight on accuracy
    y: float = 0.0
    z: float = 0.3
    
    rewardfn: Reward = Reward(min_acc=min_acc, min_size=min_size, x=x, y=y, z=z)
    ranks: tuple = get_ranks(get_model())
        
    print(f"Length of ranks: {len(ranks)}")
    print(f"Initial Model Size: {_get_model_size(get_model())}MB")
    print('-'*20)
    initial_accuracy = validate_model(get_model())[0]
    print(f'Initial Validation Accuracy: {initial_accuracy:.2f}%')
    print('-'*20)
    
    # Improved CMA-ES parameters
    es: cma.EvolutionStrategy = cma.CMAEvolutionStrategy(
        [0.25],  # Start with 25% pruning as initial guess
        0.1,     # Increased initial step size for better exploration
        {
            'bounds': [0.0, 0.9],  # Limit maximum pruning to 90%
            'maxiter': 30,
            'tolx': 1e-3,          # Convergence criteria
            'popsize': 6,          # Smaller population for faster iterations
        }
    )
    
    best_reward: float = float('-inf')
    best_pruning_amount: float = None
    history = {'pruning': [], 'rewards': [], 'accuracy': []}
    
    _clear_memory()
    iteration = 1
    while not es.stop():
        print(f"\n===== Iteration {iteration} =====")
        solutions = es.ask()
        rewards: list = []
        accuracies: list = []
        print("Evaluating solutions...")
        for x in solutions:
            pruning_amount = x[0]
            print("Evaluating pruning amount:", pruning_amount)
            reward = get_Reward(pruning_amount, ranks, rewardfn)
            accuracy = validate_model(get_pruned_model(ranks=ranks, model=get_model(), pruning_amount=pruning_amount))[0]
            print(f"Reward for {pruning_amount}: {reward}, Accuracy: {accuracy:.2f}%")
            rewards.append(-reward)  # CMA-ES minimizes, so we negate
            accuracies.append(accuracy)
            history['pruning'].append(pruning_amount)
            history['rewards'].append(reward)
            history['accuracy'].append(accuracy)
            
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = pruning_amount
            _clear_memory()
            
        es.tell(solutions, rewards)
        
        print(f"Iteration {iteration} summary:")
        print(f"- Mean pruning amount: {np.mean([s[0] for s in solutions]):.4f}")
        print(f"- Mean accuracy: {np.mean(accuracies):.2f}%")
        print(f"- Best pruning amount so far: {best_pruning_amount:.4f}")
        print(f"- Best reward so far: {best_reward:.2f}")
        print(f"- CMA-ES sigma: {es.sigma:.4f}")
        
        # Check if we're converging around min_acc
        if any(abs(acc - min_acc) < 2.0 for acc in accuracies):
            print("Found solution near target accuracy!")
        
        iteration += 1
        _clear_memory()
    
    print(f"Search complete! Best pruning amount: {best_pruning_amount:.4f}, Best reward: {best_reward:.2f}")
    
    # Final evaluation of best solution
    final_model = get_pruned_model(ranks=ranks, model=get_model(), pruning_amount=best_pruning_amount)
    final_accuracy, final_time, final_size = validate_model(final_model)
    print(f"Final evaluation - Accuracy: {final_accuracy:.2f}%, Time: {final_time:.4f}s, Size: {final_size:.2f}MB")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.scatter(history['pruning'], history['accuracy'])
    plt.axhline(y=min_acc, color='r', linestyle='--', label=f'Target Accuracy ({min_acc}%)')
    plt.xlabel('Pruning Amount')
    plt.ylabel('Accuracy (%)')
    plt.title('Pruning Amount vs Accuracy')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.scatter(history['pruning'], history['rewards'])
    plt.xlabel('Pruning Amount')
    plt.ylabel('Reward')
    plt.title('Pruning Amount vs Reward')
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.tight_layout()
    plt.savefig(f'pruning_search_results_{timestamp}.png')
    plt.close()
    
if __name__ == '__main__':
    main()
    _clear_memory()    
    print("Done")