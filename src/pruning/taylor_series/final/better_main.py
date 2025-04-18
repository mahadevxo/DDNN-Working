import gc
import torch
import numpy as np
from MVCNN.models.MVCNN import SVCNN
from MVCNN_Trainer import MVCNN_Trainer
from PruningFineTuner import PruningFineTuner
from Pruning import Pruning

def get_org_model(device='cpu'):
    model = SVCNN('svcnn')
    model.load_state_dict(torch.load('./model-00030.pth', map_location=device))
    model = model.to(device)
    return model

def init_csv():
    csv_filename = 'pruning_results_better.csv'
    with open(csv_filename, 'w', newline='') as file:
        file.write("pruning_percentage, pre_acc, post_acc, old_size, new_size\n")
    print(f"Results file created: {csv_filename} and initialized")
    
def write_to_csv(pruning_percentage, pre_acc, post_acc, old_size, new_size):
    csv_filename = 'pruning_results_better.csv'
    with open(csv_filename, 'a') as file:
        file.write(f"{pruning_percentage},{pre_acc},{post_acc},{old_size},{new_size}\n")


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    pruning_amounts = np.arange(0, 100, 1)
    
    init_csv()
    
    pft = PruningFineTuner(get_org_model(device), train_amt=0.01, test_amt=0.01)
    mvcnntrainer  = MVCNN_Trainer(optimizer=torch.optim.Adam(get_org_model(device).parameters(), lr=0.001), train_amt=0.01, test_amt=0.01)
    
    print(f"Original Model Size: {mvcnntrainer.get_size(get_org_model(device))}, Total Number of Filters: {mvcnntrainer.get_num_filters(get_org_model(device))}")
    old_size = mvcnntrainer.get_size(get_org_model(device))
    
    ranks = pft.prune(rank_filters=True)
    
    del pft
    del mvcnntrainer
    
    for pruning_amount in pruning_amounts:
        
        pre_acc = post_acc = new_size = 0.0
        
        model = get_org_model(device)
        pft = PruningFineTuner(model, train_amt=0.01, test_amt=0.1)
        num_filters_to_remove = int((pruning_amount / 100) * pft.total_num_filters())
        prune_targets = ranks[1][:num_filters_to_remove]
        
        print(f"Prune Targets: {prune_targets} for Pruning Amount {pruning_amount}")
        
        pruner = Pruning(model)
        
        for layer_index, filter_index in prune_targets:
            model = pruner.prune_conv_layers(model, layer_index=layer_index, filter_index=filter_index)
        
        mvcnntrainer = MVCNN_Trainer(optimizer=torch.optim.Adam(model.parameters(), lr=0.001), train_amt=0.1, test_amt=0.1)
        
        pre_acc = mvcnntrainer.get_val_accuracy(model)[1]
        
        model, post_acc = mvcnntrainer.fine_tune(model)
        new_size = mvcnntrainer.get_size(model)
        
        write_to_csv(pruning_amount, pre_acc, post_acc, old_size, new_size)
        print(f"Pruning {pruning_amount}: pre_acc = {pre_acc}, post_acc = {post_acc}, old_size = {old_size}, new_size = {new_size}")
        
        del model
        del mvcnntrainer
        del pruner
        torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        gc.collect()

if __name__ == '__main__':
    print("We starting")
    main()