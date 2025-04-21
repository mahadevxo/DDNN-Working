import gc
from models import MVCNN
import torch
from prune import get_ranks, get_pruned_model
from train import validate_model, fine_tune
from Rewards import Reward
import cma

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
    base_model   = get_model()                                                        # original
    pruned_model = get_pruned_model(ranks=ranks, model=base_model, pruning_amount=pruning_amount)
    pruned_model = pruned_model.to(device)
    print(f"Filters of Pruned Model: {_get_num_filters(pruned_model)}")
    finetuned_model, _ = fine_tune(pruned_model, rank_filter=False)
    accuracy, time, model_size = validate_model(finetuned_model)
    
    print('-'*20)
    
    print(f"Accuracy: {accuracy:.2f}%, Time: {time:.2f}s, Model Size: {model_size:.4f}MB")
    del base_model, pruned_model, finetuned_model
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
    x: float = 0.7
    y: float = 0.0
    z: float = 0.3
    
    rewardfn: Reward = Reward(min_acc=min_acc, min_size=min_size, x=x, y=y, z=z)
    ranks: tuple = get_ranks(get_model())
        
    print(f"Length of ranks: {len(ranks)}")
    
    es: cma.EvolutionStrategy = cma.CMAEvolutionStrategy([0.20], 0.05, {'bounds': [0.0, 1.0], 'maxiter': 30})
    best_reward: float = float('-inf')
    best_pruning_amount: float = None
    _clear_memory()
    while not es.stop():
        solutions = es.ask()
        rewards: list = []
        print("Evaluating solutions...")
        for x in solutions:
            pruning_amount = x[0]
            print("Evaluating pruning amount:", pruning_amount)
            reward = get_Reward(pruning_amount, ranks, rewardfn)
            print(f"Reward for {pruning_amount}: {reward}")
            rewards.append(-reward)
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = pruning_amount
            _clear_memory()
            del(solutions, rewards)
        es.tell(solutions, rewards)
        print("Best pruning amount so far:", best_pruning_amount)
        print("Best reward so far:", best_reward)
        print("Sigma Value:", es.sigma)
        _clear_memory()
    print(f"Best pruning amount: {best_pruning_amount}, Best reward: {best_reward}")
if __name__ == '__main__':
    main()
    _clear_memory()    
    print("Done")