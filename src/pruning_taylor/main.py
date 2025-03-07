import torchvision.models as models
from SearchAlgorithm import SearchAlgorithm
from PruningFineTuner import PruningFineTuner
import torch
import argparse

def taylor_pruning():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    pruning_fine_tuner = PruningFineTuner(model)
    out = pruning_fine_tuner.test(model)
    print(f"Original Accuracy: {out[0]:.2f}")
    print(f'Original Model Size: {pruning_fine_tuner.get_model_size(model):.2f} MB')
    del pruning_fine_tuner
    min_acc = float(input("Enter minimum acceptable accuracy (0-100): "))
    print(f"Minimum Accuracy: {min_acc}%")
    searching_strategy = SearchAlgorithm(model, min_accuracy=(min_acc/100))
    best_percentage = searching_strategy.heuristic_binary_search()
    print(f"Recommended pruning percentage: {best_percentage:.2f}%")

# New function: taylor pruning with gradient flow analysis
def taylor_pruning_with_gradient():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    # Create and register gradient flow analyzer
    from GradientFlowAnalyzer import GradientFlowAnalyzer
    analyzer = GradientFlowAnalyzer(model)
    analyzer.register_hooks()

    pruning_fine_tuner = PruningFineTuner(model)
    out = pruning_fine_tuner.test(model)
    print(f"Original Accuracy: {out[0]:.2f}")
    print(f'Original Model Size: {pruning_fine_tuner.get_model_size(model)::.2f} MB')

    # Analyze and display gradient flow statistics safely
    grad_stats = analyzer.analyze_vanishing_gradients()
    print("Gradient Flow Analysis:")
    avg_grad = grad_stats.get("average")
    if isinstance(avg_grad, (float, int)):
        print(f"  - Average Gradient: {avg_grad:.6f}")
    else:
        print("  - Average Gradient: N/A")
    median_grad = grad_stats.get("median")
    if isinstance(median_grad, (float, int)):
        print(f"  - Median Gradient: {median_grad:.6f}")
    else:
        print("  - Median Gradient: N/A")
    print(f"  - Gradient Risk: {grad_stats.get('risk', 'N/A')}")
    if small_layers := grad_stats.get("small_layers"):
        print(f"  - Layers with low gradients: {small_layers}")

    min_acc = float(input("Enter minimum acceptable accuracy (0-100): "))
    print(f"Minimum Accuracy: {min_acc}%")
    from SearchAlgorithm import SearchAlgorithm
    searching_strategy = SearchAlgorithm(model, min_accuracy=(min_acc/100))
    best_percentage = searching_strategy.heuristic_binary_search()
    print(f"Recommended pruning percentage: {best_percentage:.2f}%")

    # Optionally, call pruning to see gradient flow effects after pruning
    # pruning_fine_tuner.prune(best_percentage)

    # Cleanup
    analyzer.remove_hooks()
    del pruning_fine_tuner

# New function: combined taylor pruning with gradient-based optimization
def combined_taylor_pruning(debug=False):
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    
    # Use the unified approach that combines gradient flow analysis with optimization
    from OptimizedPruner import OptimizedPruner
    print("Initializing combined pruner...")
    combined_pruner = OptimizedPruner(model)
    
    if debug:
        print("Debug mode enabled - running with minimal operations")
        # In debug mode, just do minimal testing to check if basic operations work
        print("Testing model evaluation...")
        out = combined_pruner.test(model)
        print(f"Original Accuracy: {out[0]:.2f}")
        print(f"Original Model Size: {combined_pruner.get_model_size(model):.2f} MB")
        
        print("Testing pruning with small percentage...")
        try:
            # Skip fine-tuning completely in debug mode to avoid hook issues
            result = combined_pruner.prune(5.0, skip_fine_tuning=True)
            print(f"Test pruning result: {result}")
        except RuntimeError as e:
            print(f"Pruning failed due to in-place modification error: {e}")
        return
    
    # Normal execution flow
    out = combined_pruner.test(model)
    print(f"Original Accuracy: {out[0]:.2f}")
    print(f"Original Model Size: {combined_pruner.get_model_size(model):.2f} MB")
    
    min_acc = float(input("Enter minimum acceptable accuracy (0-100): "))
    print(f"Minimum Accuracy: {min_acc}%")
    
    from SearchAlgorithm import SearchAlgorithm
    searching_strategy = SearchAlgorithm(model, min_accuracy=(min_acc/100))
    best_percentage = searching_strategy.heuristic_binary_search_2()
    print(f"Recommended pruning percentage: {best_percentage:.2f}%")
    
if __name__ == '__main__':
    # Add command line arguments for debug mode
    parser = argparse.ArgumentParser(description='Model pruning tools')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with minimal operations')
    args = parser.parse_args()
    
    # Provide a simple menu to choose the pruning strategy
    print("Select pruning strategy:")
    print("1. Standard Taylor Pruning")
    print("2. Taylor Pruning with Gradient Flow Analysis")
    print("3. Combined Taylor Pruning with Gradient-Based Optimization")
    choice = input("Enter 1, 2 or 3: ").strip()
    if choice == '3':
        combined_taylor_pruning(debug=args.debug)
    elif choice == '2':
        taylor_pruning_with_gradient()
    else:
        taylor_pruning()