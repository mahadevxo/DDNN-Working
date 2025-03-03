import os
import sys
import torch
import torchvision.models as models
import argparse
import time
import json
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our pruning modules
from higher_order.EnhancedPruning import EnhancedPruning
from PruningFineTuner import PruningFineTuner
from higher_order.GradientFlowAnalyzer import GradientFlowAnalyzer
from higher_order.EnhancedFilterPruner import EnhancedFilterPruner

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run enhanced pruning experiments')
    
    parser.add_argument('--model', type=str, default='vgg16', 
                        choices=['vgg16', 'vgg19', 'resnet50'],
                        help='Model architecture to use')
    
    parser.add_argument('--pruning-percentages', type=float, nargs='+', 
                        default=[15.0, 30.0, 45.0, 59.0],
                        help='Pruning percentages to evaluate')
    
    parser.add_argument('--taylor-orders', type=int, nargs='+', 
                        default=[1, 2, 3],
                        help='Taylor expansion orders to evaluate')
    
    parser.add_argument('--data-path', type=str, default='imagenet-mini',
                        help='Path to dataset')
    
    parser.add_argument('--output-dir', type=str, 
                        default='results/enhanced_pruning',
                        help='Directory to save results')
    
    parser.add_argument('--fine-tune-epochs', type=int, default=5,
                        help='Number of epochs for fine-tuning')
    
    parser.add_argument('--use-gradient-flow', action='store_true',
                        help='Use gradient flow analysis')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and evaluation')
    
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Compare with baseline pruning')
    
    return parser.parse_args()

def get_model(model_name):
    """Load a pre-trained model"""
    if model_name == 'vgg16':
        return models.vgg16(pretrained=True)
    elif model_name == 'vgg19':
        return models.vgg19(pretrained=True)
    elif model_name == 'resnet50':
        return models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def create_output_directory(args):
    """Create output directory for results"""
    base_path = os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(base_path, exist_ok=True)
    
    # Create a timestamped directory for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base_path, f"{args.model}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def save_results(output_dir, results):
    """Save results to file"""
    # Save as JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot accuracy vs pruning percentage
    plot_results(output_dir, results)

def plot_results(output_dir, results):
    """Generate plots from results"""
    plt.figure(figsize=(12, 8))

    # Plot accuracy vs pruning percentage for different Taylor orders
    for taylor_order in {result['taylor_order'] for result in results}:
        order_results = [r for r in results if r['taylor_order'] == taylor_order]

        # Sort by pruning percentage
        order_results.sort(key=lambda x: x['pruning_percentage'])

        percentages = [r['pruning_percentage'] for r in order_results]
        top1_accuracies = [r['post_fine_tuning']['accuracy'][1] for r in order_results]

        plt.plot(percentages, top1_accuracies, marker='o', 
                 label=f'Taylor Order {taylor_order}')

    plt.title('Top-1 Accuracy vs Pruning Percentage')
    plt.xlabel('Pruning Percentage (%)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_pruning.png'))

    # Plot model size reduction
    plt.figure(figsize=(12, 8))

    for taylor_order in {result['taylor_order'] for result in results}:
        order_results = [r for r in results if r['taylor_order'] == taylor_order]

        # Sort by pruning percentage
        order_results.sort(key=lambda x: x['pruning_percentage'])

        percentages = [r['pruning_percentage'] for r in order_results]
        size_reductions = [r['post_fine_tuning']['size_reduction_percent'] for r in order_results]

        plt.plot(percentages, size_reductions, marker='o', 
                 label=f'Taylor Order {taylor_order}')

    plt.title('Model Size Reduction vs Pruning Percentage')
    plt.xlabel('Pruning Percentage (%)')
    plt.ylabel('Size Reduction (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'size_vs_pruning.png'))

def run_baseline_pruning(model, pruning_percentage, args):
    """Run baseline pruning using PruningFineTuner"""
    print(f"\n=== Running baseline pruning at {pruning_percentage}% ===")
    
    # Create a deep copy of the model for baseline pruning
    import copy
    baseline_model = copy.deepcopy(model)
    
    # Initialize the pruner
    pruner = PruningFineTuner(baseline_model)
    
    # Set the data paths
    pruner.train_path = os.path.join(args.data_path, 'train')
    pruner.test_path = os.path.join(args.data_path, 'val')
    
    # Run pruning
    start_time = time.time()
    baseline_results = pruner.prune(pruning_percentage)
    end_time = time.time()
    
    # Extract and format results
    pre_fine_tuning = baseline_results[0]
    accuracy = baseline_results[1]
    compute_time = baseline_results[2]
    model_size = baseline_results[3]
    
    results = {
        'method': 'baseline',
        'pruning_percentage': pruning_percentage,
        'time_taken': end_time - start_time,
        'accuracy': accuracy * 100,  # Convert to percentage
        'model_size_mb': model_size,
        'compute_time': compute_time
    }
    
    # Clean up memory
    pruner.reset()
    
    return results

def run_enhanced_pruning(model, pruning_percentage, taylor_order, use_gradient_flow, args):
    """Run enhanced pruning with specified configuration"""
    print(f"\n=== Running enhanced pruning at {pruning_percentage}% with Taylor order {taylor_order} ===")
    
    # Create a deep copy of the model
    import copy
    test_model = copy.deepcopy(model)
    
    # Initialize the enhanced pruner
    pruner = EnhancedPruning(test_model)
    
    # Run the complete pruning pipeline
    start_time = time.time()
    _, pruning_stats = pruner.run_full_pruning(
        test_model, 
        pruning_percentage, 
        taylor_order=taylor_order, 
        use_gradient_flow=use_gradient_flow,
        fine_tune_epochs=args.fine_tune_epochs
    )
    end_time = time.time()
    
    # Add timing information
    pruning_stats['time_taken'] = end_time - start_time
    pruning_stats['pruning_percentage'] = pruning_percentage
    pruning_stats['taylor_order'] = taylor_order
    pruning_stats['use_gradient_flow'] = use_gradient_flow
    
    return pruning_stats

def run_experiments(args):
    """Run all experiments based on command line arguments"""
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the pre-trained model
    model = get_model(args.model)
    model = model.to(device)
    
    # Create output directory
    output_dir = create_output_directory(args)
    
    # Save experiment configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Results storage
    all_results = []
    
    # Baseline comparison if requested
    if args.compare_baseline:
        for pruning_percentage in args.pruning_percentages:
            baseline_results = run_baseline_pruning(model, pruning_percentage, args)
            all_results.append(baseline_results)
    
    # Run enhanced pruning with different configurations
    for pruning_percentage in args.pruning_percentages:
        for taylor_order in args.taylor_orders:
            # Run with specified gradient flow setting
            pruning_stats = run_enhanced_pruning(
                model, pruning_percentage, taylor_order, 
                args.use_gradient_flow, args
            )
            all_results.append(pruning_stats)
    
    # Save and plot results
    save_results(output_dir, all_results)
    
    return all_results

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Print experiment setup
    print("=== Enhanced Pruning Experiments ===")
    print(f"Model: {args.model}")
    print(f"Pruning percentages: {args.pruning_percentages}")
    print(f"Taylor orders: {args.taylor_orders}")
    print(f"Use gradient flow: {args.use_gradient_flow}")
    print(f"Fine-tuning epochs: {args.fine_tune_epochs}")
    print(f"Dataset path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if dataset exists
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')
    
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"Warning: Dataset not found at {args.data_path}")
        print("Please ensure the dataset is in the correct location.")
        user_input = input("Continue anyway? (y/n): ").lower()
        if user_input != 'y':
            print("Exiting...")
            return
    
    # Run all experiments
    results = run_experiments(args)
    
    print("\n=== Experiments complete ===")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
