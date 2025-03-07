import torch
import numpy as np
import time
import gc

class ModelValidator:
    """Utility class to validate model integrity during pruning operations"""
    
    @staticmethod
    def validate_model(model, device='cuda'):
        """Test if the model can process a forward pass with a dummy input"""
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                output = model(dummy_input)
            return True, None
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def check_layer_sizes(model):
        """Check if any layers have invalid dimensions"""
        issues = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.in_channels <= 0:
                    issues.append(f"Layer {name} has {module.in_channels} input channels")
                if module.out_channels <= 0:
                    issues.append(f"Layer {name} has {module.out_channels} output channels")
        return issues
    
    @staticmethod
    def fix_model_layers(model, device='cuda'):
        """Attempt to fix any problematic layers in the model"""
        fixed_issues = 0
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Fix input channels
                if module.in_channels <= 0:
                    old_in = module.in_channels
                    # Create new weight tensor with at least 1 channel
                    new_in_channels = 1
                    new_weight = torch.zeros(
                        module.out_channels, 
                        new_in_channels, 
                        module.kernel_size[0], 
                        module.kernel_size[1],
                        device=device
                    )
                    # Replace the module with a fixed version
                    new_conv = torch.nn.Conv2d(
                        in_channels=new_in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=1,  # Reset groups to 1 since we're changing channels
                        bias=(module.bias is not None)
                    ).to(device)
                    # Copy over the bias if present
                    if module.bias is not None:
                        new_conv.bias.data = module.bias.data
                    # Set the new weight tensor
                    new_conv.weight.data = new_weight
                    # Replace the module
                    setattr(model, name.split('.')[-1], new_conv)
                    fixed_issues += 1
                    
                # Fix output channels
                if module.out_channels <= 0:
                    old_out = module.out_channels
                    # Create new weight tensor with at least 1 output channel
                    new_out_channels = 1
                    new_weight = torch.zeros(
                        new_out_channels, 
                        module.in_channels, 
                        module.kernel_size[0], 
                        module.kernel_size[1],
                        device=device
                    )
                    # Replace the module with a fixed version
                    new_conv = torch.nn.Conv2d(
                        in_channels=module.in_channels,
                        out_channels=new_out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        bias=(module.bias is not None)
                    ).to(device)
                    # Create new bias if needed
                    if module.bias is not None:
                        new_conv.bias.data = torch.zeros(new_out_channels, device=device)
                    # Set the new weight tensor
                    new_conv.weight.data = new_weight
                    # Replace the module
                    setattr(model, name.split('.')[-1], new_conv)
                    fixed_issues += 1
        
        return fixed_issues
    
    @staticmethod
    def analyze_model(model):
        """Print detailed analysis of model architecture"""
        print("\nModel Architecture Analysis:")
        total_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                params = sum(p.numel() for p in module.parameters())
                total_params += params
                
                if isinstance(module, torch.nn.Conv2d):
                    print(f"{name}: Conv2d(in={module.in_channels}, out={module.out_channels}, "
                          f"kernel={module.kernel_size}, stride={module.stride}, "
                          f"padding={module.padding}), params={params:,}")
                
                if isinstance(module, torch.nn.Linear):
                    print(f"{name}: Linear(in={module.in_features}, out={module.out_features}), "
                          f"params={params:,}")
        
        print(f"\nTotal parameters: {total_params:,}")
        return total_params

class PruningDebugger:
    """Debug utils specifically for pruning operations"""
    
    @staticmethod
    def debug_pruning(model, layer_index, filter_index, pruner_class):
        """Debug a specific pruning operation"""
        print(f"\n=== Debugging pruning operation: layer {layer_index}, filter {filter_index} ===")
        
        # Analyze the layer before pruning
        modules = list(model.features._modules.items())
        if layer_index < len(modules):
            _, layer = modules[layer_index]
            if isinstance(layer, torch.nn.Conv2d):
                print(f"Original layer: in_channels={layer.in_channels}, out_channels={layer.out_channels}")
            else:
                print(f"Layer at index {layer_index} is not Conv2d: {type(layer)}")
        else:
            print(f"Layer index {layer_index} out of range (max: {len(modules)-1})")
            return
            
        # Initialize pruner
        pruner = pruner_class(model)
        
        # Try pruning with detailed error handling
        try:
            # Make a clone of the model to avoid modifying the original during debug
            model_copy = type(model)()
            model_copy.load_state_dict(model.state_dict())
            
            # Try the pruning operation
            new_model = pruner.prune_vgg_conv_layer(model_copy, layer_index, filter_index)
            
            # Validate pruned model
            valid, error = ModelValidator.validate_model(new_model)
            if valid:
                print("Pruning operation successful!")
                # Check the modified layer
                modules = list(new_model.features._modules.items())
                if layer_index < len(modules):
                    _, layer = modules[layer_index]
                    if isinstance(layer, torch.nn.Conv2d):
                        print(f"Pruned layer: in_channels={layer.in_channels}, out_channels={layer.out_channels}")
            else:
                print(f"Pruned model validation failed: {error}")
                issues = ModelValidator.check_layer_sizes(new_model)
                if issues:
                    print("Layer size issues:")
                    for issue in issues:
                        print(f"  - {issue}")
        except Exception as e:
            import traceback
            print(f"Error during pruning: {str(e)}")
            traceback.print_exc()
