import gc
from models import MVCNN
import torch
from prune import get_ranks, get_pruned_model
from train import validate_model, fine_tune
from Rewards import Reward
import cma

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_model():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MVCNN.SVCNN('SVCNN')
    weights = torch.load('./model-00030.pth', map_location=device)
    model.load_state_dict(weights)
    model = model.to(device)
    del weights
    _clear_memory()
    return model

def get_Reward(pruning_amount, ranks, rewardfn):
    model = get_model()
    model = get_pruned_model(ranks=ranks, model=model, pruning_amount=pruning_amount)
    model = model.to(device)
    model, _ = fine_tune(model, rank_filter=False)
    accuracy, time, model_size = validate_model(model)
    print(f"Accuracy: {accuracy:.2f}%, Time: {time:.2f}s, Model Size: {model_size:.2f}MB")
    del model
    reward = rewardfn.get_reward(accuracy, time, model_size, )
    _clear_memory()
    return reward

def search():
    # res = validate_model(get_model())
    # print(f"Initial Validation Accuracy: {res[0]:.2f}%, Time: {res[1]:.6f}s, Model Size: {res[2]:.2f}MB")
    min_acc = 50
    min_size = 300
    x=0.7
    y=0.0
    z=0.3
    
    rewardfn = Reward(min_acc=min_acc, min_size=min_size, x=x, y=y, z=z)
    ranks = get_ranks(get_model())
        
    print(f"Length of ranks: {len(ranks)}")
    
    es = cma.CMAEvolutionStrategy([0.15], 0.05, {'bounds': [0.0, 1.0]})
    best_reward = float('-inf')
    best_pruning_amount = None
    
    while not es.stop():
        solutions = es.ask()
        rewards = []
        
        for x in solutions:
            pruning_amount = x[0]
            print("Evaluating pruning amount:", pruning_amount)
            reward = get_Reward(pruning_amount, ranks, rewardfn)
            print(f"Reward for {pruning_amount}: {reward}")
            rewards.append(-reward)
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = pruning_amount
            es.tell(solutions, rewards)
        print("Best pruning amount so far:", best_pruning_amount)
        print("Best reward so far:", best_reward)
        print("Sigma Value:", es.sigma)
    
    print(f"Best pruning amount: {best_pruning_amount}, Best reward: {best_reward}")

if __name__ == '__main__':
    search()