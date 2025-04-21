import torch
import sys
import os
import json
from models import MVCNN
from train import get_detailed_model_info

def analyze_model_structure(model):
    """Produce a detailed analysis of the model's structure and parameter distribution"""
    print("=" * 50)
    print("DETAILED MODEL ANALYSIS")
    print("=" * 50)
    
    # Get basic model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Get detailed breakdown by module type
    info = get_detailed_model_info(model)
    
    # Print parameter distribution by layer type
    print("\nParameter distribution by layer type:")
    print(f"  - Conv layers: {info['conv_params']:,} ({info['conv_params']/total_params*100:.1f}%)")
    print(f"  - Linear layers: {info['linear_params']:,} ({info['linear_params']/total_params*100:.1f}%)")
    print(f"  - BatchNorm layers: {info['bn_params']:,} ({info['bn_params']/total_params*100:.1f}%)")
    print(f"  - Other layers: {info['other_params']:,} ({info['other_params']/total_params*100:.1f}%)")
    
    # Print breakdown of other module types
    if info['other_module_types']:
        print("\nBreakdown of 'Other' module types:")
        for module_type, params in sorted(info['other_module_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {module_type}: {params:,} parameters ({params/total_params*100:.1f}%)")
    
    # Print untracked parameters
    if info['untracked_params'] > 0:
        print(f"\nUntracked parameters: {info['untracked_params']:,} ({info['untracked_params']/total_params*100:.1f}%)")
    
    # Analyze model hierarchy
    print("\nStructure of large parameter blocks:")
    _print_large_param_blocks(info['model_hierarchy'], threshold=0.01*total_params)
    
    # List largest individual layers
    print("\nLargest individual layers:")
    for i, layer in enumerate(sorted(info['layer_info'], key=lambda x: x['params'], reverse=True)[:10]):
        print(f"  {i+1}. {layer['name']} ({layer['type']}): {layer['params']:,} parameters")
        if 'shape' in layer:
            print(f"     Shape: {layer['shape']}")
    
    # Check for common inefficiencies
    print("\nPotential inefficiencies:")
    _check_for_inefficiencies(model, info)
    
    print("\n" + "=" * 50)

def _print_large_param_blocks(hierarchy, prefix="", depth=0, threshold=1000, max_depth=3):
    """Recursively print the hierarchy of parameter blocks exceeding the threshold"""
    if depth > max_depth:
        return
    
    for name, content in hierarchy.items():
        if isinstance(content, dict):
            # This is a sub-module
            params = _count_params_in_hierarchy(content)
            if params > threshold:
                print(f"{'  ' * depth}- {prefix + ('.' if prefix else '')}{name}: {params:,} parameters")
                _print_large_param_blocks(content, f"{prefix + ('.' if prefix else '')}{name}", depth+1, threshold, max_depth)
        else:
            # This is a parameter
            if content > threshold:
                print(f"{'  ' * depth}- {prefix + ('.' if prefix else '')}{name}: {content:,} elements")

def _count_params_in_hierarchy(hierarchy):
    """Count total parameters in a hierarchy dictionary"""
    count = 0
    for name, content in hierarchy.items():
        if isinstance(content, dict):
            count += _count_params_in_hierarchy(content)
        else:
            count += content
    return count

def _check_for_inefficiencies(model, info):
    """Check for common inefficiencies in the model architecture"""
    # Check for duplicate sequential layers
    seq_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Sequential):
            seq_layers.append((name, len(list(module.children()))))
    
    if any(count > 10 for _, count in seq_layers):
        print("  - Found large Sequential blocks which might be inefficient")
        for name, count in seq_layers:
            if count > 10:
                print(f"    - {name}: {count} sub-modules")
    
    # Check for high parameter redundancy in linear layers
    total_linear = info['linear_params']
    if total_linear > info['total_params'] * 0.8:
        print("  - Linear layers consume >80% of parameters, consider reducing their size")
    
    # Check for unused parameters
    if info['untracked_params'] > info['total_params'] * 0.01:
        print(f"  - {info['untracked_params']:,} parameters aren't tracked in modules (could be unused)")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model
    model_path = './model-00030.pth'  # Default path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    print(f"Loading model from: {model_path}")
    model = MVCNN.SVCNN('SVCNN')
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
    model = model.to(device)
    
    # Analyze the model
    analyze_model_structure(model)
    
    # Save a detailed report to a JSON file
    info = get_detailed_model_info(model)
    with open('model_analysis_report.json', 'w') as f:
        # Convert any non-serializable parts
        serializable_info = {k: v for k, v in info.items() if k != 'model_hierarchy'}
        json.dump(serializable_info, f, indent=2)
    
    print(f"Detailed analysis saved to model_analysis_report.json")
