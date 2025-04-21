import gc
from heapq import nsmallest
from operator import itemgetter
import torch
import copy
from train import get_train_data, get_model_size_by_params, get_detailed_model_info  # add get_train_data, get_model_size_by_params, get_detailed_model_info
from FilterPruner import FilterPruner
from Pruning import Pruning
from itertools import groupby

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
def _get_sorted_filters(ranks):    
    if ranks is None:
        raise ValueError("Ranks must be provided")
    data = []
    for i in sorted(ranks.keys()):
        for j in range(ranks[i].size(0)):
            data.append((i, j, ranks[i][j]))
    _clear_memory()
    return sorted(data, key=lambda x: x[2])

def get_ranks(model):
    model_copy = copy.deepcopy(model) 
    pruner = FilterPruner(model_copy)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = get_train_data(train_amt=0.05)
    print(f"train_loader: {len(train_loader)}")

    model_copy.eval()
    for data in train_loader:
        in_data = data[1].to(device)
        labels = data[0].to(device)
        output = pruner.forward(in_data)
        loss = criterion(output, labels)
        loss.backward()
    # normalize + retrieve the actual dict
    ranks_dict = pruner.normalize_ranks_per_layer()
    ranks = pruner.get_sorted_filters(ranks_dict)
    _clear_memory()
    del model_copy
    del pruner
    _clear_memory()
    # Debug: check if ranks are empty
    if not ranks:
        print("Warning: Ranks are empty after computation.")
        return None
    return ranks

def _get_pruning_plan(num, ranks):
    if ranks is None:
        raise ValueError("Ranks must be provided")
    num = int(num)
    if num == 0:
        return None
    
    filters_to_prune = nsmallest(num, ranks, key=itemgetter(2))
    print(f"Pruning {len(filters_to_prune)} filters")
    
    filters_to_prune_per_layer = {}
    
    for (layer_n, filter, _) in filters_to_prune:
        if layer_n not in filters_to_prune_per_layer:
            filters_to_prune_per_layer[layer_n] = []
        filters_to_prune_per_layer[layer_n].append(filter)
    
    for layer_n in filters_to_prune_per_layer:
        filters_to_prune_per_layer[layer_n] = sorted(filters_to_prune_per_layer[layer_n])
        for i in range(len(filters_to_prune_per_layer[layer_n])):
            filters_to_prune_per_layer[layer_n][i] = int(filters_to_prune_per_layer[layer_n][i]) - i
    
    filters_to_prune = []
    for layer_n in filters_to_prune_per_layer:
        for filter in filters_to_prune_per_layer[layer_n]:
            filters_to_prune.append((layer_n, filter))
    
    _clear_memory()
    return filters_to_prune

def _prune_model(prune_targets, model):
    
    for param in model.net_1.parameters():
        param.requires_grad = True
    
    pruner = Pruning(model)
    
    # group targets by layer, prune highest indices first to avoid shifting
    sorted_targets = sorted(prune_targets, key=lambda x: (x[0], -x[1]))
    groups = groupby(sorted_targets, key=lambda x: x[0])
    for layer_n, group in groups:
        group = sorted(list(group), key=lambda x: -x[1])  # ensure descending filter indices
        for _, filter_index in group:
            pruner = Pruning(model)  # reinitialize with current model
            model = pruner.prune_conv_layers(model=model, layer_index=layer_n, filter_index=filter_index)
    
    print(f"Pruned {len(prune_targets)} filters")
    _clear_memory()
    
    # After pruning, ensure model is properly cleaned up
    # This ensures abandoned parameters are removed from memory
    model = copy.deepcopy(model)
    
    # Convert all parameters to float32 to ensure consistent size calculation
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            if layer.weight is not None:
                layer.weight.data = layer.weight.data.float()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.float()
    
    # Log parameter counts before and after pruning
    model_info = get_detailed_model_info(model)
    print(f"After pruning: {model_info['total_params']:,} parameters ({model_info['model_size_mb']:.2f} MB)")
    print(f"  - Conv: {model_info['conv_params']:,}, Linear: {model_info['linear_params']:,}")
    
    del pruner
    
    model = model.to(device)
    # Explicitly call garbage collector again to clean up any lingering tensors
    _clear_memory()
    return model

def get_pruned_model(ranks=None, model=None, pruning_amount=0.0):
    model_copy = copy.deepcopy(model)
    
    # Calculate and log model size before pruning
    initial_size = get_model_size_by_params(model_copy)
    initial_info = get_detailed_model_info(model_copy)
    
    if ranks is None:
        ranks = get_ranks(model_copy)
    try:
        # Count filters before pruning
        total_filters = sum(m.out_channels for m in model_copy.net_1 if isinstance(m, torch.nn.Conv2d))
    except Exception as e:
        print(f"Error calculating total filters: {e}")
        exit()
        
    print(f"Before pruning: {initial_info['total_params']:,} parameters ({initial_size:.2f} MB)")
    print(f"Total conv filters before: {total_filters}, pruning_amount: {pruning_amount*100:.1f}%")
    
    num_filters_to_prune = int(pruning_amount * total_filters)
    prune_targets = _get_pruning_plan(num_filters_to_prune, ranks)
    model_pruned = _prune_model(prune_targets, model_copy)
    
    # Count filters and size after pruning
    new_total = sum(m.out_channels for m in model_pruned.net_1 if isinstance(m, torch.nn.Conv2d))
    final_size = get_model_size_by_params(model_pruned)
    
    print(f"Total conv filters after: {new_total} (removed {total_filters-new_total} filters)")
    print(f"Model size: {initial_size:.2f}MB → {final_size:.2f}MB ({(1-final_size/initial_size)*100:.1f}% reduction)")
    
    # Clean memory
    del model_copy
    _clear_memory()
    
    return model_pruned