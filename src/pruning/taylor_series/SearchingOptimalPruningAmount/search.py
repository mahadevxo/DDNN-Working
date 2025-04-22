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
    return (
        load_model(device)
        if '_cached_model' not in globals()
        else copy.deepcopy(_cached_model)
    )


def load_model(device):
    print("Loading model from disk...")
    result: torch.nn.Module = MVCNN.SVCNN('SVCNN')
    weights: dict = torch.load('./model-00030.pth', map_location=device)
    result.load_state_dict(weights)
    result: torch.nn.Module = result.to(device)
    _cached_model = result
    del weights
    _clear_memory()
    return result

def get_Reward(pruning_amount: float, ranks: tuple, rewardfn: Reward, rank_type='taylor', min_acc=None) -> tuple:
    global device
    
    # If min_acc is not provided, use a default value
    if min_acc is None:
        min_acc = 50.0
    
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
        print("Model parameter distribution:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - net_1 parameters: {net1_params:,} ({net1_params/total_params*100:.1f}%)")
        print(f"  - Conv parameters: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
        print(f"  - Linear parameters: {linear_params:,} ({linear_params/total_params*100:.1f}%)")
        del base_model
        _clear_memory()
    
    # Skip extreme pruning amounts that are likely to fail
    if pruning_amount > 0.85:
        print(f"Skipping excessive pruning amount: {pruning_amount}")
        return -10000  # Return a highly negative reward
    
    # Create pruned model with adapter and specified rank type
    base_model = get_model()
    pruned_model = get_pruned_model(ranks=ranks, model=base_model, 
                                    pruning_amount=pruning_amount, 
                                    adapt_interface=True, 
                                    adapter_mode='zero_pad',
                                    rank_type=rank_type)
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
    reward, comp_time = rewardfn.getReward(accuracy=accuracy, comp_time=time, 
                                           model_size=model_size, comp_time_last=comp_time_last, 
                                           param_reduction=param_reduction)
    comp_time_last = comp_time
    
    # Add aggressive bonus for high pruning rates with good accuracy
    if accuracy >= min_acc and pruning_amount >= 0.4:
        bonus = pruning_amount * 200  # More aggressive bonus for high pruning rates
        reward += bonus
        print(f"Added high pruning bonus: +{bonus:.2f}")
    
    print(f"Reward: {reward}")
    print('-'*20)
    
    return reward

def main() -> None:  # sourcery skip: low-code-quality
    # Use fewer iterations and smarter exploration
    min_acc: float = 50.0
    min_size: float = 30.0
    x: float = 0.6  # Slightly lower weight on accuracy to encourage more pruning
    y: float = 0.1  # Add some weight to size
    z: float = 0.3

    # Choose ranking method
    rank_type = 'taylor'  # Fall back to taylor since combined is causing issues
    print(f"Using {rank_type} ranking criterion for pruning")

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
    try:
        ranks: tuple = get_ranks(get_model(), rank_type=rank_type)
        if ranks is None or not ranks:
            print("Warning: Failed to compute ranks, falling back to taylor criterion")
            rank_type = 'taylor'
            ranks = get_ranks(get_model(), rank_type=rank_type)
    except Exception as e:
        print(f"Error computing ranks with {rank_type} criterion: {e}")
        print("Falling back to taylor criterion")
        rank_type = 'taylor'
        ranks = get_ranks(get_model(), rank_type=rank_type)

    print(f"Length of ranks: {len(ranks)}")
    print('-'*20)

    # More aggressive CMA-ES parameters
    try:
        es: cma.EvolutionStrategy = cma.CMAEvolutionStrategy(
            [0.05],  # Start with higher pruning rate
            0.25,    # Much larger step size for aggressive exploration
            {
                'bounds': [0.0005, 0.85],  # Allow more extreme pruning (excluding 0)
                'maxiter': 12,           # Slightly more iterations to find optimum
                'tolx': 1e-2,            # Tighter convergence for precision
                'popsize': 6,            # Larger population for better exploration
                'verbose': 1,            # Reduce verbosity
            }
        )
    except Exception as e:
        print(f"Error initializing CMA-ES: {e}")
        print("Falling back to simpler initialization")
        # Fallback to a simpler but still aggressive initialization
        es = cma.CMAEvolutionStrategy(
            [0.05],
            0.25,  
            {'bounds': [0.05, 0.85], 'verbose': 1}
        )

    best_reward: float = float('-inf')
    best_pruning_amount: float = None
    history = {'pruning': [], 'rewards': [], 'accuracy': []}
    
    # Track multiple promising regions with their rewards for better search
    promising_regions = []
    stagnation_counter = 0
    prev_best_reward = float('-inf')
    
    # First perform aggressive binary search across the entire range
    print("\n===== Binary Search Exploration =====")
    binary_search_points = [0.1, 0.25, 0.4, 0.55, 0.7]
    binary_rewards = []
    
    for point in binary_search_points:
        print(f"Binary search point: {point:.4f}")
        reward = get_Reward(point, ranks, rewardfn, rank_type=rank_type, min_acc=min_acc)
        binary_rewards.append((point, reward))
        
        history['pruning'].append(point)
        history['rewards'].append(reward)
        
        if reward > best_reward:
            best_reward = reward
            best_pruning_amount = point
            print(f"New best pruning amount: {best_pruning_amount:.4f} (reward: {best_reward:.2f})")
    
    # Sort by reward and select top regions for further exploration
    binary_rewards.sort(key=lambda x: x[1], reverse=True)
    
    # Add more focused exploration points around the best regions
    for i, (point, reward) in enumerate(binary_rewards[:2]):  # Focus on top 2 regions
        width = 0.15  # Wider exploration around promising points
        min_bound = max(0.05, point - width/2)
        max_bound = min(0.85, point + width/2)
        
        # Explore each promising region with denser sampling
        for offset in [-0.07, -0.03, 0.03, 0.07]:
            new_point = point + offset
            if min_bound <= new_point <= max_bound:
                promising_regions.append((max(0.05, new_point - 0.05), 
                                         min(0.85, new_point + 0.05)))
    
    # For early exploration of fixed points with adaptive density
    if best_pruning_amount is not None:
        # Create denser exploration around the current best point
        best_region_min = max(0.05, best_pruning_amount - 0.12)
        best_region_max = min(0.85, best_pruning_amount + 0.12)
        initial_points = np.linspace(best_region_min, best_region_max, 5)
    else:
        # If no good point found, explore the full range with bias toward likely regions
        initial_points = [0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.7]
    
    print("\n===== Initial Targeted Exploration =====")
    for point in initial_points:
        print(f"Evaluating initial point: {point:.4f}")
        reward = get_Reward(point, ranks, rewardfn, rank_type=rank_type, min_acc=min_acc)
        
        history['pruning'].append(point)
        history['rewards'].append(reward)
        
        if reward > best_reward:
            best_reward = reward
            best_pruning_amount = point
            print(f"New best pruning amount: {best_pruning_amount:.4f} (reward: {best_reward:.2f})")
            
            # Mark smaller region around good points for fine-grained search
            promising_regions.append((max(0.05, point - 0.05), min(0.85, point + 0.05)))
    
    _clear_memory()
    iteration = 1
    max_iterations = 15  # Cap on iterations to prevent wasting time on unproductive search
    improvement_threshold = 5.0  # Threshold for considering a significant improvement

    # More aggressive exploration of promising regions
    def explore_region(center, width=0.1, steps=7):  # More steps for finer search
        nonlocal best_reward, best_pruning_amount
        print(f"Exploring promising region around {center:.4f}...")
        
        min_bound = max(0.05, center - width/2)
        max_bound = min(0.85, center + width/2)
        
        # Use non-uniform sampling with more points near center
        points = []
        points.append(center)  # Always include the center
        
        # Add points with higher density near the center
        for step in range(1, (steps//2) + 1):
            offset = width * step/(steps-1) * 0.8  # 0.8 factor to concentrate points
            points.append(center - offset)
            points.append(center + offset)
        
        for point in sorted(points):
            if min_bound <= point <= max_bound:
                print(f"  Trying fine-grained point: {point:.4f}")
                reward = get_Reward(point, ranks, rewardfn, rank_type=rank_type, min_acc=min_acc)
                
                history['pruning'].append(point)
                history['rewards'].append(reward)
                
                if reward > best_reward:
                    best_reward = reward
                    best_pruning_amount = point
                    print(f"New best pruning amount: {best_pruning_amount:.4f} (reward: {best_reward:.2f})")
                    
                    # For very good points, search even more aggressively around them
                    if reward > prev_best_reward + improvement_threshold:
                        return True  # Signal that we found a significantly better point
        
        _clear_memory()
        return False

    while not es.stop() and iteration <= max_iterations:
        print(f"\n===== Iteration {iteration} =====")
        try:
            solutions = es.ask()
            rewards: list = []

            print("Evaluating solutions...")
            for x in solutions:
                pruning_amount = x[0]
                print(f"Evaluating pruning amount: {pruning_amount:.4f}")

                # Get reward includes validation - pass the rank_type
                reward = get_Reward(pruning_amount, ranks, rewardfn, rank_type=rank_type, min_acc=min_acc)
                rewards.append(-reward)  # CMA-ES minimizes, so we negate

                # Track the best solution found
                if reward > best_reward:
                    best_reward = reward
                    best_pruning_amount = pruning_amount
                    print(f"New best pruning amount: {best_pruning_amount:.4f} (reward: {best_reward:.2f})")
                    
                    # If we find a significantly better solution, explore around it immediately
                    if reward > prev_best_reward + 5:  # Only if significant improvement
                        explore_region(pruning_amount, width=0.08, steps=3)
                    
                    # Mark this region for further exploration
                    promising_regions.append((max(0.0, pruning_amount - 0.1), min(0.8, pruning_amount + 0.1)))

                # Store history
                history['pruning'].append(pruning_amount)
                history['rewards'].append(reward)

                _clear_memory()

            # Handle empty solutions list or other edge cases
            if not solutions or not rewards:
                print("Warning: No valid solutions or rewards in this iteration")
                break

            try:
                es.tell(solutions, rewards)
                
                # Check for stagnation
                if abs(prev_best_reward - best_reward) < 1e-3:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                prev_best_reward = best_reward
                
                # If stagnating, either explore promising regions or restart with new parameters
                if stagnation_counter >= 2:
                    print("Search stagnating, trying alternative approaches...")
                    
                    # First, explore any promising regions we've found
                    if promising_regions:
                        region = promising_regions.pop(0)
                        center = (region[0] + region[1]) / 2
                        width = region[1] - region[0]
                        explore_region(center, width, steps=5)
                        stagnation_counter = 0  # Reset counter after exploration
                    else:
                        # If no promising regions or already explored, restart with larger step size
                        print("Restarting search with more aggressive parameters...")
                        if best_pruning_amount is not None:
                            es = cma.CMAEvolutionStrategy(
                                [best_pruning_amount],  # Start from best known
                                0.15,  # Large step size
                                {'bounds': [0.0, 0.8], 'verbose': 1}
                            )
                        stagnation_counter = 0
                
            except ValueError as e:
                print(f"Error in CMA-ES tell method: {e}")
                print("Trying an alternative approach...")

                # If we encounter dimension issues, try a manual grid search instead
                if iteration == 1:  # Only do this if we fail on first iteration
                    print("Falling back to grid search")
                    grid_pruning_amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    grid_rewards = []

                    for amount in grid_pruning_amounts:
                        print(f"Grid search - evaluating pruning amount: {amount:.2f}")
                        reward = get_Reward(amount, ranks, rewardfn, rank_type=rank_type, min_acc=min_acc)
                        grid_rewards.append(reward)

                        if reward > best_reward:
                            best_reward = reward
                            best_pruning_amount = amount

                    # We're done with grid search, exit the loop
                    print("Grid search complete")
                else:
                    # Just continue with what we have
                    print("Continuing with best results so far")
                break
        except Exception as e:
            print(f"Error in iteration {iteration}: {e}")
            print("Continuing with best results so far")
            break

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
    if best_pruning_amount is None:
        best_pruning_amount = 0.25  # Fallback to a reasonable default
        print(f"Using default pruning amount of {best_pruning_amount} as no valid solution was found")

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
    # show best pruning amount and reward
    plt.axvline(x=best_pruning_amount, color='r', linestyle='--', label=f'Best Pruning Amount: {best_pruning_amount:.4f}')
    plt.axhline(y=best_reward, color='g', linestyle='--', label=f'Best Reward: {best_reward:.2f}')
    plt.legend()
    plt.grid()
    plt.savefig(f'pruning_search_results_{timestamp}.png')
    plt.close()
    
if __name__ == '__main__':
    main()
    _clear_memory()    
    print("Done")