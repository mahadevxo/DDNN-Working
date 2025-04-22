import gc
import time
import torch
import numpy as np
from FilterPruner import FilterPruner
from tools.ImgDataset import SingleImgDataset

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def _get_num_filters(model: torch.nn.Module) -> int:
    return sum(
        module.out_channels
        for _, module in model.net_1._modules.items()
        if isinstance(module, torch.nn.Conv2d)
    )

def get_model_size(model: torch.nn.Module, only_net_1: bool = False) -> float:
    """Calculate model size based on parameter count"""
    if only_net_1:
        total_params = sum(p.numel() for p in model.net_1.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params * 4 / (1024 ** 2)  # Convert to MB (assuming float32)

def get_model_size_by_params(model: torch.nn.Module, only_net_1: bool = False) -> float:
    """More accurate model size calculation that accounts for parameter data types"""
    total_bytes = 0
    
    # Handle cases where model structure might differ
    if only_net_1:
        # Try different ways to access net_1 parameters
        if hasattr(model, 'net_1'):
            params_iterator = model.net_1.parameters()
        elif hasattr(model, 'model_1'):  # For adapter models
            params_iterator = model.model_1.parameters()
        elif hasattr(model, 'adapter') and hasattr(model.adapter, 'model_1'):
            params_iterator = model.adapter.model_1.parameters()
        else:
            print("Warning: Could not find net_1, calculating size for entire model")
            params_iterator = model.parameters()
    else:
        params_iterator = model.parameters()
        
    for p in params_iterator:
        if p.requires_grad:
            # Account for actual parameter size based on data type
            if p.dtype == torch.float32 or p.dtype not in [
                torch.float16,
                torch.int8,
            ]:
                bytes_per_param = 4
            elif p.dtype == torch.float16:
                bytes_per_param = 2
            else:
                bytes_per_param = 1
            total_bytes += p.numel() * bytes_per_param

    # Add estimated overhead for model structure (typically ~10%)
    total_bytes *= 1.1
    return total_bytes / (1024 ** 2)  # Convert to MB

def get_detailed_model_info(model: torch.nn.Module) -> dict:
    # sourcery skip: low-code-quality
    """Get detailed information about model parameters and size by layer"""
    conv_params = 0
    linear_params = 0
    bn_params = 0  # Batch normalization parameters
    other_params = 0
    other_module_types = {}  # Track what other module types exist
    
    layer_info = []
    
    # Counter for parameters not captured in named_modules
    untracked_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            conv_params += params
            untracked_params -= params
            layer_info.append({
                'name': name, 
                'type': 'Conv2d',
                'params': params,
                'shape': f"({module.in_channels}, {module.out_channels}, {module.kernel_size[0]}, {module.kernel_size[1]})"
            })
        elif isinstance(module, torch.nn.Linear):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            linear_params += params
            untracked_params -= params
            layer_info.append({
                'name': name, 
                'type': 'Linear',
                'params': params,
                'shape': f"({module.in_features}, {module.out_features})"
            })
        elif isinstance(module, torch.nn.BatchNorm2d):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            bn_params += params
            untracked_params -= params
            layer_info.append({
                'name': name,
                'type': 'BatchNorm2d',
                'params': params,
                'shape': f"(channels: {module.num_features})"
            })
        elif any(p.requires_grad for p in module.parameters()):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                other_params += params
                module_type = type(module).__name__
                if module_type not in other_module_types:
                    other_module_types[module_type] = 0
                other_module_types[module_type] += params
                layer_info.append({
                    'name': name,
                    'type': module_type,
                    'params': params
                })
                # Don't subtract from untracked_params here as we might double count

    # Analyze model hierarchy to find any loose parameters
    model_hierarchy = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            parts = name.split('.')
            current = model_hierarchy
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            if parts[-1] not in current:
                current[parts[-1]] = param.numel()
    
    total_params = conv_params + linear_params + bn_params + other_params
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    return {
        'total_params': total_params,
        'conv_params': conv_params,
        'linear_params': linear_params,
        'bn_params': bn_params,
        'other_params': other_params,
        'other_module_types': other_module_types,
        'untracked_params': untracked_params,
        'model_hierarchy': model_hierarchy,
        'model_size_mb': model_size_mb,
        'layer_info': layer_info
    }

def get_train_data(train_path: str='ModelNet40-12View/*/train', train_amt: float=0.05, num_models: int=1000, num_views: int=12) -> torch.utils.data.DataLoader:
    classes_present = []
    train_dataset = SingleImgDataset(
        train_path, scale_aug=False, rot_aug=False,
        num_models=num_models, num_views=num_views,
    )
    total_models = len(train_dataset.filepaths) // num_views
    # Determine how many models to sample
    subset_size = int(train_amt * total_models)

    # Randomly sample model indices
    rand_model_indices = np.random.permutation(total_models)[:subset_size]

    new_filepaths = []
    for idx in rand_model_indices:
        start = idx * num_views
        end = (idx + 1) * num_views
        new_filepaths.extend(train_dataset.filepaths[start:end])

    # Assign the new filepaths to the dataset
    train_dataset.filepaths = new_filepaths
    for filepath in train_dataset.filepaths:
        class_name = filepath.split('/')[3]
        if class_name not in classes_present:
            classes_present.append(class_name)
    
    if len(classes_present) < 33:
        print(f"Subset of classes present: {len(classes_present)} out of 33")
        return False
    _clear_memory()
    return torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=4
    )

def get_test_data(test_path: str='ModelNet40-12View/*/test', num_models: int=1000, num_views: int=12) -> torch.utils.data.DataLoader:
    test_dataset = SingleImgDataset(
        test_path, scale_aug=False, rot_aug=False,
        num_models=num_models, num_views=num_views,
    )
    _clear_memory()
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

def analyze_network_compatibility(model):
    """Check if the dimensions between net_1 output and net_2 input are compatible"""
    net_1_output_channels = next(
        (
            module.out_channels
            for module in reversed(list(model.net_1))
            if isinstance(module, torch.nn.Conv2d)
        ),
        0,
    )
    net_2_input_size = next(
        (
            module.in_features
            for module in model.net_2
            if isinstance(module, torch.nn.Linear)
        ),
        None,
    )
    # Detect spatial dimensions through a dummy forward pass
    spatial_size = None
    try:
        dummy_input = torch.zeros(1, 3, 224, 224).to(device)
        with torch.no_grad():
            net_1_output = model.net_1(dummy_input)
            _, _, h, w = net_1_output.shape
            spatial_size = (h, w)
            print(f"Network interface - net_1 output shape: {net_1_output.shape}")
    except Exception as e:
        print(f"Could not detect spatial dimensions: {e}")
        # Use default spatial size as fallback
        spatial_size = (7, 7)

    print(f"Network interface - net_1 output channels: {net_1_output_channels}")
    print(f"Network interface - detected spatial size: {spatial_size}")

    if net_2_input_size:
        print(f"Network interface - net_2 input features: {net_2_input_size}")
    else:
        print("Could not determine net_2 input size - no Linear layer found")

    # Calculate the expected interface size
    calculated_interface = net_1_output_channels * spatial_size[0] * spatial_size[1]

    if net_2_input_size:
        if calculated_interface != net_2_input_size:
            print("Warning: Dimension mismatch at network interface!")
            print(f"  - Calculated size: {calculated_interface}")
            print(f"  - Expected input size for net_2: {net_2_input_size}")
            print(f"  - Difference: {abs(calculated_interface - net_2_input_size)} features")
        else:
            print("Network interface dimensions are compatible")

    return {
        "net_1_output_channels": net_1_output_channels,
        "net_2_input_size": net_2_input_size,
        "spatial_size": spatial_size,
        "calculated_interface": calculated_interface,
        "is_compatible": calculated_interface == net_2_input_size
    }

def get_model_structure_info(model):
    """Get detailed information about model parts: net_1 and net_2"""
    # Analyze net_1 (feature extractor)
    net_1_info = {
        'conv_layers': 0,
        'bn_layers': 0,
        'parameters': 0,
        'last_conv_output_channels': 0
    }
    
    for module in model.net_1:
        if isinstance(module, torch.nn.Conv2d):
            net_1_info['conv_layers'] += 1
            net_1_info['last_conv_output_channels'] = module.out_channels
            net_1_info['parameters'] += sum(p.numel() for p in module.parameters() if p.requires_grad)
        elif isinstance(module, torch.nn.BatchNorm2d):
            net_1_info['bn_layers'] += 1
            net_1_info['parameters'] += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Analyze net_2 (classifier)
    net_2_info = {
        'linear_layers': 0,
        'parameters': 0,
        'first_linear_input_features': 0
    }
    
    for module in model.net_2:
        if isinstance(module, torch.nn.Linear):
            if net_2_info['linear_layers'] == 0:
                net_2_info['first_linear_input_features'] = module.in_features
            net_2_info['linear_layers'] += 1
            net_2_info['parameters'] += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    return {
        'net_1': net_1_info,
        'net_2': net_2_info
    }

def validate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader=None, only_net_1_size: bool = False) -> tuple:
    if test_loader is None:
        test_loader = get_test_data()
    
    # Check model structure and compatibility between parts
    structure_info = get_model_structure_info(model)
    _ = analyze_network_compatibility(model)
    
    print("\nModel Structure:")
    print(f"  - net_1: {structure_info['net_1']['conv_layers']} conv layers, "
          f"{structure_info['net_1']['bn_layers']} bn layers, "
          f"{structure_info['net_1']['parameters']:,} parameters")
    print(f"  - net_2: {structure_info['net_2']['linear_layers']} linear layers, "
          f"{structure_info['net_2']['parameters']:,} parameters")
    print("")
    
    all_correct_points = 0
    all_point = 0
    all_loss = 0
    wrong_class = np.zeros(33)
    samples_class = np.zeros(33)
    times = []
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            
            in_data = data[1].to(device)
            labels = data[0].to(device)
            
            t1 = time.time()
            output = model(in_data)
            t2 = time.time()
            times.append(t2 - t1)
            
            pred = torch.max(output, 1)[1]
            all_loss += torch.nn.CrossEntropyLoss()(output, labels).cpu().detach().numpy()
            results = pred==labels
            
            for i in range(results.size()[0]):
                if not bool(results[i].cpu().numpy()):
                    wrong_class[labels[i].cpu().numpy()] += 1
                else:
                    all_correct_points += 1
                all_point += 1
                samples_class[labels[i].cpu().numpy()] += 1
    all_loss /= len(test_loader)
    
    validation_accuracy = (all_correct_points / all_point) * 100
    times = np.mean(times)
    
    # Enhanced model size calculation
    model_size_in_mb = get_model_size_by_params(model, only_net_1=only_net_1_size)
    
    # Print detailed model information with improved formatting
    model_info = get_detailed_model_info(model)
    net1_params = sum(p.numel() for p in model.net_1.parameters() if p.requires_grad)
    net1_size_mb = net1_params * 4 / (1024 ** 2)  # Assuming float32
    
    print(f"Model parameters: {model_info['total_params']:,} ({model_info['model_size_mb']:.2f} MB)")
    print(f"  - net_1 parameters: {net1_params:,} ({net1_size_mb:.2f} MB, {net1_params/model_info['total_params']*100:.1f}%)")
    print(f"  - Conv layers: {model_info['conv_params']:,} params ({model_info['conv_params']/model_info['total_params']*100:.1f}%)")
    print(f"  - Linear layers: {model_info['linear_params']:,} params ({model_info['linear_params']/model_info['total_params']*100:.1f}%)")
    print(f"  - BatchNorm layers: {model_info['bn_params']:,} params ({model_info['bn_params']/model_info['total_params']*100:.1f}%)")
    print(f"  - Other layers: {model_info['other_params']:,} params ({model_info['other_params']/model_info['total_params']*100:.1f}%)")
    
    # Print any untracked parameters
    if model_info['untracked_params'] > 0:
        print(f"  - Untracked parameters: {model_info['untracked_params']:,}")
    
    # Print breakdown of other module types if any exist
    if model_info['other_module_types']:
        print("  Other module types breakdown:")
        for module_type, count in sorted(model_info['other_module_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {module_type}: {count:,} params ({count/model_info['total_params']*100:.1f}%)")
    
    del model, test_loader
    _clear_memory()
    return validation_accuracy, times, model_size_in_mb

def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader=None, 
               rank_filter: bool=False, train_amt: float=0.1) -> torch.nn.Module:
    # Update to accept train_amt parameter
    model = model.to(device)
    model = model.train()
    
    if rank_filter:
        for param in model.net_1.parameters():
            param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    if train_loader is None:
        while True:
            x = get_train_data(train_amt=train_amt)
            if x is not False:
                train_loader = x
                break
            else:
                print("Not enough classes in the training set. Retrying...")
    
    running_loss, running_accc, total_steps = 0.0, 0.0, 0.0
    
    if rank_filter:
        pruner = FilterPruner(model)
    
    for batch_idx, data in enumerate(train_loader):
        try:
            with torch.autograd.set_grad_enabled(True):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    model.zero_grad(set_to_none=True)
                
                in_data = data[1].to(device)
                labels = data[0].to(device)
                
                output = pruner.forward(in_data) if rank_filter else model(in_data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                pred = torch.max(output, 1)[1]
                results = pred == labels
                correct_points = torch.sum(results.long())
                acc = correct_points.float() / results.size()[0]
                running_accc += acc.item()
                total_steps += 1
        except Exception as exp:
            print(f"Error in batch {batch_idx}: {exp}")
            continue
    running_loss /= total_steps
    running_accc /= total_steps
    print(f"Training loss: {running_loss:.3f}, Training accuracy: {running_accc:.3f}")
    
    _clear_memory()
    if rank_filter:
        del pruner
    del train_loader, optimizer, criterion
    return model

def fine_tune(model: torch.nn.Module, rank_filter: bool=False, quick_mode: bool=False) -> tuple:
    """Fine-tune the model with option for quick evaluation mode"""
    print(f"Fine Tuning Model; Model has {_get_num_filters(model)} filters")
    model = model.to(device)
    model = model.train()
    
    print('-----Getting Stats-----')
    val_acc, times, _ = validate_model(model)
    print(f'Initial validation - Accuracy: {val_acc:.2f}%, Time: {times:.6f}s')
    
    model = model.train()
    epoch = 0
    prev_accs = []
    best_accuracy = val_acc
    
    # In quick mode, do fewer epochs for faster evaluation
    max_epochs = 5 if quick_mode else 10
    early_stop_threshold = 3.0 if quick_mode else 6.0
    
    while True:
        print(f"--------Epoch {epoch+1}--------")
        model = train_model(model, train_amt=0.03 if quick_mode else 0.1)
        accuracy = validate_model(model)[0]
        prev_accs.append(accuracy)
        
        if len(prev_accs) > 3:
            prev_accs.pop(0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"Best accuracy so far: {best_accuracy:.2f}%")
        
        # Early stopping when accuracy stabilizes
        if epoch > 1 and abs(best_accuracy - np.mean(prev_accs).item()) < early_stop_threshold:
            print(f"Stopping fine-tuning at epoch {epoch+1} - accuracy stabilized")
            break
        
        if epoch >= (max_epochs - 1):
            print(f"Max epochs reached ({max_epochs})")
            break
        
        print(f"Epoch {epoch+1} -> Validation accuracy: {accuracy:.2f}%")
        epoch += 1
    
    print(f"Final validation accuracy: {accuracy:.2f}%")
    
    _clear_memory()
    return model, accuracy