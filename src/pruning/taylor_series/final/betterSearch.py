import gc
import torch
import numpy as np
from MVCNN.models.MVCNN import SVCNN
from MVCNN_Trainer import MVCNN_Trainer
from PruningFineTuner import PruningFineTuner
from Pruning import Pruning
from Rewards import Reward
from bayes_opt import BayesianOptimization

def get_model_org():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SVCNN('svcnn')
    model.load_state_dict(torch.load('./model-00030.pth'))
    model = model.to(device)
    return model

def init_csv():
    csv_filename = 'search_results.csv'
    with open(csv_filename, 'w', newline='') as file:
        file.write("pruning_amount,reward\n")
    print(f"Results file created: {csv_filename} and initialized")
    
def write_to_csv(pruning_amount, reward):
    csv_filename = 'search_results.csv'
    with open(csv_filename, 'a', newline='') as file:
        file.write(f"{pruning_amount},{reward}\n")

def prune_model(ranks, pruning_amount, comp_times, x, y, z):
    rewardFn = Reward(min_acc=60.0, min_size=300.0)
    
    model = get_model_org()
    
    pft = PruningFineTuner(model, train_amt=0.1, test_amt=0.5)
    num_filters_to_remove = int((pruning_amount / 100) * pft.total_num_filters())
    prune_targets = ranks[1][:num_filters_to_remove]
    
    print(f"Prune Targets: {len(prune_targets)} for Pruning Amount {pruning_amount}")
    
    pruner = Pruning(model)
    
    for layer_index, filter_index in prune_targets:
        model = pruner.prune_conv_layers(model, layer_index=layer_index, filter_index=filter_index)
    
    mvcnntrainer = MVCNN_Trainer(optimizer=torch.optim.Adam(model.parameters(), lr=0.001), train_amt=0.1, test_amt=0.5)
    pre_acc = mvcnntrainer.test(model)
    new_size = mvcnntrainer.get_size(model)
    
    model, post_acc = mvcnntrainer.fine_tune(model)
    comp_time = mvcnntrainer.get_val_accuracy(model)[3]
    del mvcnntrainer
    del model
    del pruner
    torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    gc.collect()
    
    reward = rewardFn.getReward(accuracy=post_acc, model_size=new_size, comp_time=comp_time, comp_time_last=comp_times, x=x, y=y, z=z)
    write_to_csv(pruning_amount=pruning_amount, reward=reward)
    return reward, comp_time
    
def something():
    model = get_model_org()
    init_csv()    
    pft = PruningFineTuner(get_model_org(), train_amt=0.1, test_amt=0.5)
    mvcnntrainer = MVCNN_Trainer(optimizer=torch.optim.Adam(get_model_org().parameters(), lr=0.001), train_amt=0.1, test_amt=0.5)
    
    print(f"Original Model Size: {mvcnntrainer.get_size(get_model_org())}, Total Number of Filters: {mvcnntrainer.get_num_filters(get_model_org())}")
    comp_time_old = mvcnntrainer.get_val_accuracy(model)[3]
    ranks = pft.prune(rank_filters=True)
    
    del model, pft, mvcnntrainer
    import gc
    gc.collect()
    
    # Define objective function for BayesianOptimization
    def objective(pruning_amount):
        reward, _ = prune_model(ranks=ranks, pruning_amount=pruning_amount,
                                  comp_times=comp_time_old, x=0.7, y=0.0, z=0.3)
        return reward

    pbounds = {'pruning_amount': (0, 100)}
    optimizer_bo = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1)
    optimizer_bo.maximize(init_points=5, n_iter=10)
    print("Best result:", optimizer_bo.max)

if __name__ == '__main__':
    something()