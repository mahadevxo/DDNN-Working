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
        for name, module in model.net_1._modules.items()
        if isinstance(module, torch.nn.Conv2d)
    )

def get_model_size(model: torch.nn.Module) -> float:
    """Calculate model size based on parameter count"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params * 4 / (1024 ** 2)  # Convert to MB (assuming float32)

def get_model_size_by_params(model: torch.nn.Module) -> float:
    """More accurate model size calculation that accounts for parameter data types"""
    total_bytes = 0
    for p in model.parameters():
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
    """Get detailed information about model parameters and size by layer"""
    conv_params = 0
    linear_params = 0
    other_params = 0
    
    layer_info = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            conv_params += params
            layer_info.append({
                'name': name, 
                'type': 'Conv2d',
                'params': params,
                'shape': f"({module.in_channels}, {module.out_channels}, {module.kernel_size[0]}, {module.kernel_size[1]})"
            })
        elif isinstance(module, torch.nn.Linear):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            linear_params += params
            layer_info.append({
                'name': name, 
                'type': 'Linear',
                'params': params,
                'shape': f"({module.in_features}, {module.out_features})"
            })
        elif any(p.requires_grad for p in module.parameters()):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                other_params += params
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': params
                })
    
    total_params = conv_params + linear_params + other_params
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
    
    return {
        'total_params': total_params,
        'conv_params': conv_params,
        'linear_params': linear_params,
        'other_params': other_params,
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

def validate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader=None) -> tuple:
    if test_loader is None:
        test_loader = get_test_data()
    
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
    model_size_in_mb = get_model_size_by_params(model)
    
    # Print detailed model information
    model_info = get_detailed_model_info(model)
    print(f"Model parameters: {model_info['total_params']:,} ({model_info['model_size_mb']:.2f} MB)")
    print(f"  - Conv layers: {model_info['conv_params']:,} params")
    print(f"  - Linear layers: {model_info['linear_params']:,} params")
    
    del model, test_loader
    _clear_memory()
    return validation_accuracy, times, model_size_in_mb

def train_model(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader=None, rank_filter: bool=False) -> torch.nn.Module:
    model = model.to(device)
    model = model.train()
    
    if rank_filter:
        for param in model.net_1.parameters():
            param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    if train_loader is None:
        while True:
            x = get_train_data()
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

def fine_tune(model: torch.nn.Module, rank_filter: bool=False) -> tuple:
    print(f"Fine Tuning Model; Model has {_get_num_filters(model)} filters")
    model = model.to(device)
    model = model.train()
    
    print('-----Getting Stats-----')
    val_acc, times, _ = validate_model(model)
    print(f'Validation time: {times:.6f}s')
    print(f'Validation accuracy: {val_acc:.2f}%')
    # print(f'Model size: {model_size:.4f}MB')
    
    model = model.train()
    epoch = 0
    prev_accs = []
    best_accuracy = 0
    
    while True:
        print(f"--------Epoch {epoch+1}--------")
        model = train_model(model)
        accuracy = validate_model(model)[0]
        prev_accs.append(accuracy)
        
        if len(prev_accs) > 5:
            prev_accs.pop(0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"Best accuracy so far: {best_accuracy:.2f}%")
        
        if epoch > 3 and best_accuracy - 2 <= np.mean(prev_accs).item() <= best_accuracy + 2:
            print(f"Stopping fine-tuning at epoch {epoch+1} with accuracy {accuracy:.2f}%")
            break
        
        if epoch >= 4:
            print(f"Max Epochs Reached-{epoch+1}")
            break
        
        print(f"Epoch {epoch+1} -> Validation accuracy: {accuracy:.2f}%")
        epoch += 1
        
    print(f"Final validation accuracy: {accuracy:.2f}%")
    print(f"Final validation time: {times:.6f}s")
    
    _clear_memory()
    return model, accuracy