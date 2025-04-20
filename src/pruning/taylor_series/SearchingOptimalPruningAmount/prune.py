import gc
from heapq import nsmallest
from operator import itemgetter
import torch
from train import get_train_data  # add get_train_data
from FilterPruner import FilterPruner
from Pruning import Pruning
import tqdm

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
    # model = model.to(device).train()
    # model = train_model(model, rank_filter=True)

    pruner = FilterPruner(model)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = get_train_data(train_amt=0.01)
    print(f"train_loader: {len(train_loader)}")

    model.eval()
    for data in train_loader:
        in_data = data[1].to(device)
        labels = data[0].to(device)
        output = pruner.forward(in_data)
        loss = criterion(output, labels)
        loss.backward()
    # now normalize
    ranks = pruner.normalize_ranks_per_layer()
    ranks  = _get_sorted_filters(ranks)
    # print(f'prune.py: normalized ranks dict -> {ranks}')
    # exit()
    _clear_memory()
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
    pruner = Pruning(model)
    for layer_n, filter_index in enumerate(tqdm(prune_targets, desc="Pruning filters")):
        model = pruner.prune_conv_layers(model=model, layer_index=layer_n, filter_index=filter_index)
    print(f"Pruned {len(prune_targets)} filters")
    _clear_memory()
    
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
            if layer.weight is not None:
                layer.weight.data = layer.weight.data.float()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.float()
    
    model = model.to(device)
    _clear_memory()
    return model

def get_pruned_model(ranks=None, model=None, pruning_amount=0.0):
    if ranks is None:
        ranks = get_ranks(model)
    try:
        total_filters = sum(
            layer.out_channels
            for layer in model.net_1
            if isinstance(layer, torch.nn.Conv2d)
        )
    except Exception as e:
        print(f"Error calculating total filters: {e}")
        exit()
    num_filters_to_prune = int(pruning_amount * total_filters)

    prune_targets = _get_pruning_plan(num_filters_to_prune, ranks)
    model = _prune_model(prune_targets, model)
    _clear_memory()
    return model