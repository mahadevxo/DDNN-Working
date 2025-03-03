import os
import sys
import torch
import torchvision.models as models
import argparse
import time
import copy

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import our pruning modules
from higher_order.EnhancedPruning import EnhancedPruning

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run a single enhanced pruning experiment')
    
    parser.add_argument('--model', type=str, default='vgg16', 
                        choices=['vgg16', 'vgg19', 'resnet50'],
                        help='Model architecture to use')
    
    parser.add_argument('--pruning-percentage', type=float, default=30.0,
                        help='Percentage of filters to prune')
    
    parser.add_argument('--taylor-order', type=int, default=2,
                        choices=[1, 2, 3],
                        help='Order of Taylor expansion')
    
    parser.add_argument('--data-path', type=str, default='imagenet-mini',
                        help='Path to dataset')
    
    parser.add_argument('--fine-tune-epochs', type=int, default=5,
                        help='Number of epochs for fine-tuning')
    
    parser.add_argument('--use-gradient-flow', action='store_true',
                        help='Use gradient flow analysis')
    
    return parser.parse_args()

def main():
    """Main entry point for single experiment"""
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"=== Enhanced Pruning Experiment ===")
    print(f"Model: {args.model}")
    print(f"Pruning percentage: {args.pruning_percentage}%")
    print(f"Taylor order: {args.taylor_order}")
    print(f"Use gradient flow: {args.use_gradient_flow}")
    print(f"Fine-tuning epochs: {args.fine_tune_epochs}")
    
    # Load pre-trained model
    if args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.model == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Initialize model on the chosen device
    model = model.to(device)
    
    # Print initial model information
    original_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Original model size: {original_size_mb:.2f} MB")
    
    # Initialize the enhanced pruner
    pruner = EnhancedPruning(model)
    
    # Run the complete pruning pipeline
    start_time = time.time()
    pruned_model, pruning_stats = pruner.run_full_pruning(
        model, 
        args.pruning_percentage, 
        taylor_order=args.taylor_order, 
        use_gradient_flow=args.use_gradient_flow,
        fine_tune_epochs=args.fine_tune_epochs
    )
    end_time = time.time()
    
    # Calculate time taken
    time_taken = end_time - start_time
    print(f"\nTime taken: {time_taken:.2f} seconds")
    
    # Print detailed results
    print("\n=== Detailed Results ===")
    print(f"Initial accuracy - Top-1: {pruning_stats['initial']['accuracy'][1]:.2f}%, "
          f"Top-5: {pruning_stats['initial']['accuracy'][2]:.2f}%")
    print(f"Accuracy after pruning - Top-1: {pruning_stats['post_pruning']['accuracy'][1]:.2f}%, "
          f"Top-5: {pruning_stats['post_pruning']['accuracy'][2]:.2f}%")
    print(f"Final accuracy - Top-1: {pruning_stats['post_fine_tuning']['accuracy'][1]:.2f}%, "
          f"Top-5: {pruning_stats['post_fine_tuning']['accuracy'][2]:.2f}%")
    print(f"Model size reduction: {pruning_stats['post_fine_tuning']['size_reduction_percent']:.2f}%")
    print(f"From {pruning_stats['initial']['size_mb']:.2f} MB to {pruning_stats['post_fine_tuning']['size_mb']:.2f} MB")
    
    # Pruned filters per layer
    print("\nPruned filters per layer:")
    for layer, count in pruning_stats['pruning_details']['filters_per_layer'].items():
        print(f"Layer {layer}: {count} filters pruned")
    
    # Save the pruned model if desired
    save_model = input("\nSave pruned model? (y/n): ").lower()
    if save_model == 'y':
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_filename = f"{args.model}_pruned_{args.pruning_percentage}pct_order{args.taylor_order}_{timestamp}.pth"
        torch.save(pruned_model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    main()
