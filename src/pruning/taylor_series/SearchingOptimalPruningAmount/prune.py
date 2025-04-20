import gc
from heapq import nsmallest
from operator import itemgetter
import torch
from train import train_model
from FilterPruner import FilterPruner
from Pruning import Pruning

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
    pruner = FilterPruner(model)
    pruner.reset()
    model = model.to(device)
    model = train_model(model, rank_filter=True) #model with ranks ready for sorting
    ranks = pruner.normalize_ranks_per_layer()
    _clear_memory()
    return _get_sorted_filters(ranks)

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
    pruner = Pruning(model)
    for idx, (layer_n, filter_index) in enumerate(prune_targets):
        model = pruner.prune_conv_layers(model=model, layer_index=layer_n, filter_index=filter_index)
        if idx % 10 == 0:
            print(f"Pruned {idx} filters")
    print(f"Pruned {len(prune_targets)} filters")
    _clear_memory()
    
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.weight.data = layer.weight.data.to(device)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(device)
    
    model = model.to(device)
    _clear_memory()
    return model

def get_pruned_model(ranks=None, model=None, pruning_amount=0.0):
    if ranks is None:
        ranks = get_ranks(model)

    total_filters = sum(len(ranks[i]) for i in ranks)
    num_filters_to_prune = int(pruning_amount * total_filters)

    prune_targets = _get_pruning_plan(num_filters_to_prune, ranks)
    model = _prune_model(prune_targets, model)
    _clear_memory()
    return model
