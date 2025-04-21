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

def get_pruned_model(ranks=None, model=None, pruning_amount=0.0, adapt_interface=True, adapter_mode='zero_pad'):
    model_copy = copy.deepcopy(model)
    
    # Calculate and log detailed model stats before pruning - focusing on net_1
    initial_size = get_model_size_by_params(model_copy, only_net_1=True)
    initial_info = get_detailed_model_info(model_copy)
    
    if ranks is None:
        ranks = get_ranks(model_copy)
    try:
        # Count filters before pruning - only in net_1
        total_filters = sum(m.out_channels for m in model_copy.net_1 if isinstance(m, torch.nn.Conv2d))
        # Count parameters in net_1 vs net_2
        net1_params = sum(p.numel() for p in model_copy.net_1.parameters() if p.requires_grad)
        net2_params = sum(p.numel() for p in model_copy.net_2.parameters() if p.requires_grad)
        conv_params = sum(p.numel() for name, m in model_copy.named_modules() 
                          if isinstance(m, torch.nn.Conv2d) 
                          for p in m.parameters() if p.requires_grad)
        linear_params = sum(p.numel() for name, m in model_copy.named_modules() 
                            if isinstance(m, torch.nn.Linear) 
                            for p in m.parameters() if p.requires_grad)
    except Exception as e:
        print(f"Error calculating model stats: {e}")
        exit()
        
    print(f"Before pruning:")
    print(f"  - net_1 size: {initial_size:.2f} MB")
    print(f"  - Total parameters: {net1_params + net2_params:,}")
    print(f"  - net_1 parameters: {net1_params:,} ({net1_params/(net1_params+net2_params)*100:.1f}%)")
    print(f"  - net_2 parameters: {net2_params:,} ({net2_params/(net1_params+net2_params)*100:.1f}%)")
    print(f"  - Conv parameters: {conv_params:,}, Linear parameters: {linear_params:,}")
    print(f"  - Total conv filters in net_1: {total_filters}, pruning_amount: {pruning_amount*100:.1f}%")
    
    num_filters_to_prune = int(pruning_amount * total_filters)
    prune_targets = _get_pruning_plan(num_filters_to_prune, ranks)
    
    # Apply model surgery - only to net_1
    model_pruned = _prune_model(prune_targets, model_copy)
    
    # Force model optimization after pruning
    model_pruned = _optimize_model_after_pruning(model_pruned)
    
    # Check network interface compatibility
    from train import analyze_network_compatibility
    compatibility = analyze_network_compatibility(model_pruned)
    
    # Handle potential interface issues between net_1 and net_2 if requested
    if adapt_interface:
        try:
            from network_adapter import AdaptedModel
            print(f"Applying network adapter with mode: {adapter_mode}")
            model_pruned = AdaptedModel(model_pruned, adapter_mode)
        except Exception as e:
            print(f"Warning: Could not apply network adapter: {e}")
    
    # Calculate detailed stats after pruning - focusing on net_1
    new_total_filters = sum(m.out_channels for m in model_pruned.net_1 if isinstance(m, torch.nn.Conv2d))
    new_net1_params = sum(p.numel() for p in model_pruned.net_1.parameters() if p.requires_grad)
    new_net2_params = sum(p.numel() for p in model_pruned.net_2.parameters() if p.requires_grad)
    new_conv_params = sum(p.numel() for name, m in model_pruned.named_modules() 
                      if isinstance(m, torch.nn.Conv2d) 
                      for p in m.parameters() if p.requires_grad)
    new_linear_params = sum(p.numel() for name, m in model_pruned.named_modules() 
                        if isinstance(m, torch.nn.Linear) 
                        for p in m.parameters() if p.requires_grad)
    final_size = get_model_size_by_params(model_pruned, only_net_1=True)
    
    print(f"After pruning:")
    print(f"  - Conv filters in net_1: {new_total_filters} (removed {total_filters-new_total_filters} filters, {(1-new_total_filters/total_filters)*100:.1f}% reduction)")
    print(f"  - net_1 parameters: {new_net1_params:,} (removed {net1_params-new_net1_params:,} params, {(1-new_net1_params/net1_params)*100:.1f}% reduction)")
    print(f"  - net_2 parameters: {new_net2_params:,} (should be unchanged)")
    print(f"  - Total params: {new_net1_params+new_net2_params:,} (removed {(net1_params+net2_params)-(new_net1_params+new_net2_params):,} params, {(1-(new_net1_params+new_net2_params)/(net1_params+net2_params))*100:.1f}% reduction)")
    print(f"  - net_1 size: {initial_size:.2f}MB → {final_size:.2f}MB ({(1-final_size/initial_size)*100:.1f}% reduction)")
    
    # Clean memory
    del model_copy
    _clear_memory()
    
    return model_pruned

def _optimize_model_after_pruning(model):
    """Ensure model is properly optimized after pruning"""
    # Create a new model instance to ensure clean state
    optimized_model = copy.deepcopy(model)
    
    # Force proper memory layout and data types for net_1
    for module in optimized_model.net_1.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.dtype == torch.float32:
                # Use contiguous memory layout for better efficiency
                module.weight.data = module.weight.data.contiguous()
        if hasattr(module, 'bias') and module.bias is not None:
            if module.bias.dtype == torch.float32:
                module.bias.data = module.bias.data.contiguous()
    
    # Test forward pass with a dummy input to verify network integrity
    try:
        dummy_input = torch.zeros(1, 3, 224, 224).to(next(optimized_model.parameters()).device)
        with torch.no_grad():
            _ = optimized_model(dummy_input)
        print("Model passed integrity check with dummy forward pass")
    except Exception as e:
        print(f"Warning: Model failed integrity check: {e}")
    
    # Ensure proper cleanup
    _clear_memory()
    
    return optimized_model