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
import copy

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

def _get_model_size(model, only_net_1=True):
    """Calculate model size more accurately including only required parameters"""
    from train import get_model_size_by_params
    
    # Add safety check for adapter models
    if only_net_1 and hasattr(model, 'adapter') and not hasattr(model, 'net_1'):
        # If this is an adapter without net_1, use the inner nets
        return get_model_size_by_params(model.adapter, only_net_1=only_net_1)
    
    return get_model_size_by_params(model, only_net_1=only_net_1)

def get_model() -> torch.nn.Module:
    """Cache the model to avoid repeated loading from disk"""
    global device, _cached_model
    if '_cached_model' not in globals():
        print("Loading model from disk...")
        model: torch.nn.Module = MVCNN.SVCNN('SVCNN')
        weights: dict = torch.load('./model-00030.pth', map_location=device)
        model.load_state_dict(weights)
        model: torch.nn.Module = model.to(device)
        _cached_model = model
        del weights
        _clear_memory()
    else:
        # Return a copy of the cached model
        model = copy.deepcopy(_cached_model)
    
    return model

def get_Reward(pruning_amount: float, ranks: tuple, rewardfn: Reward) -> tuple:
    global device
    
    # Only compute the original size once - focusing on net_1
    if not hasattr(get_Reward, 'original_size'):
        base_model = get_model()
        get_Reward.original_size = _get_model_size(base_model, only_net_1=True)
        get_Reward.original_filters = _get_num_filters(base_model)
        # Calculate parameter distribution to understand the model better
        total_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        net1_params = sum(p.numel() for p in base_model.net_1.parameters() if p.requires_grad)
        conv_params = sum(p.numel() for name, m in base_model.named_modules() 
                      if isinstance(m, torch.nn.Conv2d) 
                      for p in m.parameters() if p.requires_grad)
        linear_params = sum(p.numel() for name, m in base_model.named_modules() 
                        if isinstance(m, torch.nn.Linear) 
                        for p in m.parameters() if p.requires_grad)
        print(f"Model parameter distribution:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - net_1 parameters: {net1_params:,} ({net1_params/total_params*100:.1f}%)")
        print(f"  - Conv parameters: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
        print(f"  - Linear parameters: {linear_params:,} ({linear_params/total_params*100:.1f}%)")
        del base_model
        _clear_memory()
    
    # Skip extreme pruning amounts that are likely to fail
    if pruning_amount > 0.85:
        print(f"Skipping excessive pruning amount: {pruning_amount}")
        return -1000  # Return a highly negative reward
    
    # Create pruned model with adapter
    base_model = get_model()
    pruned_model = get_pruned_model(ranks=ranks, model=base_model, 
                                    pruning_amount=pruning_amount, 
                                    adapt_interface=True, 
                                    adapter_mode='zero_pad')
    pruned_model = pruned_model.to(device)
    pruned_size = _get_model_size(pruned_model, only_net_1=True)
    pruned_filters = _get_num_filters(pruned_model)
    
    # Calculate parameter reduction - focusing on net_1
    base_net1_params = sum(p.numel() for p in base_model.net_1.parameters() if p.requires_grad)
    pruned_net1_params = sum(p.numel() for p in pruned_model.net_1.parameters() if p.requires_grad)
    param_reduction = (1 - pruned_net1_params/base_net1_params) * 100
    
    print(f"Filters: {get_Reward.original_filters} → {pruned_filters} ({pruned_filters/get_Reward.original_filters*100:.1f}%)")
    print(f"net_1 Parameters: {base_net1_params:,} → {pruned_net1_params:,} ({param_reduction:.1f}% reduction)")
    print(f"net_1 Size: {get_Reward.original_size:.2f}MB → {pruned_size:.2f}MB ({(1 - pruned_size/get_Reward.original_size)*100:.1f}% reduction)")
    
    # Only fine-tune models that have a reasonable chance of success
    if pruned_filters < get_Reward.original_filters * 0.1:
        print("Too few filters remain, skipping fine-tuning")
        del base_model, pruned_model
        _clear_memory()
        return -2000
    
    # Fine-tune with limited epochs for quick evaluation
    finetuned_model, accuracy = fine_tune(pruned_model, quick_mode=True)
    _, time, model_size = validate_model(finetuned_model, only_net_1_size=True)
    
    print(f"Accuracy: {accuracy:.2f}%, Time: {time:.2f}s, Model Size: {model_size:.4f}MB")
    
    # Force cleanup of models
    del base_model, pruned_model, finetuned_model
    _clear_memory()
    
    global comp_time_last
    if comp_time_last is None:
        comp_time_last = time
    
    # Adjust reward calculation to better account for parameter reduction
    reward, comp_time = rewardfn.getReward(accuracy=accuracy, comp_time=time, model_size=model_size, comp_time_last=comp_time_last, param_reduction=param_reduction)
    comp_time_last = comp_time
    
    print(f"Reward: {reward}")
    print('-'*20)
    
    return reward

def main() -> None:
    # Use fewer iterations and smarter exploration
    min_acc: float = 50
    min_size: float = 300
    x: float = 0.7  # Higher weight on accuracy
    y: float = 0.0
    z: float = 0.3
    
    print("Initial Model Info:")
    base_model = get_model()
    initial_size = _get_model_size(base_model, only_net_1=True)
    initial_filters = _get_num_filters(base_model)
    initial_accuracy = validate_model(base_model)[0]
    print(f"net_1 Size: {initial_size:.2f}MB, Filters: {initial_filters}, Accuracy: {initial_accuracy:.2f}%")
    del base_model
    _clear_memory()
    
    rewardfn: Reward = Reward(min_acc=min_acc, min_size=min_size, x=x, y=y, z=z)
    
    # Cache the ranks to avoid recomputing
    print("Computing filter ranks...")
    ranks: tuple = get_ranks(get_model())
    print(f"Length of ranks: {len(ranks)}")
    print('-'*20)
    
    # More efficient CMA-ES parameters
    es: cma.EvolutionStrategy = cma.CMAEvolutionStrategy(
        [0.25],  # Start with 25% pruning as initial guess
        0.1,     # Increased initial step size for better exploration
        {
            'bounds': [0.0, 0.8],  # Limit maximum pruning to 80%
            'maxiter': 10,         # Reduce max iterations
            'tolx': 2e-2,          # More relaxed convergence criteria
            'popsize': 4,          # Smaller population
            'verbose': 1,          # Reduce verbosity
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
        
        print("Evaluating solutions...")
        for x in solutions:
            pruning_amount = x[0]
            print(f"Evaluating pruning amount: {pruning_amount:.4f}")
            
            # Get reward includes validation
            reward = get_Reward(pruning_amount, ranks, rewardfn)
            rewards.append(-reward)  # CMA-ES minimizes, so we negate
            
            # Track the best solution found
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = pruning_amount
                print(f"New best pruning amount: {best_pruning_amount:.4f} (reward: {best_reward:.2f})")
            
            # Store history
            history['pruning'].append(pruning_amount)
            history['rewards'].append(reward)
            
            _clear_memory()
            
        es.tell(solutions, rewards)
        
        print(f"Iteration {iteration} summary:")
        print(f"- Mean pruning amount: {np.mean([s[0] for s in solutions]):.4f}")
        print(f"- Best pruning amount so far: {best_pruning_amount:.4f}")
        print(f"- Best reward so far: {best_reward:.2f}")
        print(f"- CMA-ES sigma: {es.sigma:.4f}")
        
        iteration += 1
        _clear_memory()
    
    print(f"Search complete! Best pruning amount: {best_pruning_amount:.4f}, Best reward: {best_reward:.2f}")
    
    # Final evaluation with the best solution (with full fine-tuning)
    print("\n===== Final Evaluation =====")
    final_model = get_pruned_model(ranks=ranks, model=get_model(), pruning_amount=best_pruning_amount)
    final_model, final_accuracy = fine_tune(final_model, quick_mode=False)
    final_time, final_size = validate_model(final_model)[1:]
    
    print(f"Final model - Accuracy: {final_accuracy:.2f}%, Time: {final_time:.4f}s, Size: {final_size:.2f}MB")
    
    # Save the final model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(final_model.state_dict(), f'pruned_model_{timestamp}.pth')
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.scatter(history['pruning'], history['rewards'])
    plt.xlabel('Pruning Amount')
    plt.ylabel('Reward')
    plt.title('Pruning Amount vs Reward')
    plt.tight_layout()
    plt.savefig(f'pruning_search_results_{timestamp}.png')
    plt.close()
    
if __name__ == '__main__':
    main()
    _clear_memory()    
    print("Done")