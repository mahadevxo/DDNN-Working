import torch
import argparse
from MVCNN.models import MVCNN
from FilterPruner import FilterPruner
from Pruning import Pruning
from PFT import PruningFineTuner
import gc

class PruningTester:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
    def _clear_memory(self):
        """Helper method to clear memory efficiently"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def load_model(self):
        """Load the SVCNN model"""
        model = MVCNN.SVCNN('svcnn')
        model.load_state_dict(torch.load('mvcnn.pth', map_location=self.device))
        model = model.to(self.device)
        return model
    
    def get_model_size(self, model):
        param_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        buffer_size = sum(
            buffer.nelement() * buffer.element_size() for buffer in model.buffers()
        )
        return (param_size + buffer_size) / 1024**2
    
    def get_layer_info(self, model):
        """Get detailed information about model layers"""
        print("\nLayer-by-layer breakdown:")
        
        # Analyze convolutional layers
        print("\nConvolutional layers:")
        for i, layer in enumerate(model.net_1):
            if isinstance(layer, torch.nn.Conv2d):
                params = sum(p.numel() for p in layer.parameters())
                print(f"  net_1[{i}] Conv2d: in={layer.in_channels}, out={layer.out_channels}, kernel={layer.kernel_size}, params={params:,}")
        
        # Analyze fully-connected layers
        print("\nFully connected layers:")
        for i, layer in enumerate(model.net_2):
            if isinstance(layer, torch.nn.Linear):
                params = sum(p.numel() for p in layer.parameters())
                print(f"  net_2[{i}] Linear: in={layer.in_features}, out={layer.out_features}, params={params:,}")
    
    def prune_model(self, model, pruning_amount=0.1):
        # sourcery skip: low-code-quality
        """Prune the model by a specified percentage"""
        print(f"\n------- Pruning model by {pruning_amount*100:.1f}% -------")

        # Analyze model before pruning
        print("\nModel BEFORE pruning:")
        total_params_before = sum(p.numel() for p in model.parameters())
        conv_params_before = sum(p.numel() for name, layer in enumerate(model.net_1) 
                               if isinstance(layer, torch.nn.Conv2d)
                               for p in layer.parameters())
        fc_params_before = sum(p.numel() for name, layer in enumerate(model.net_2) 
                             if isinstance(layer, torch.nn.Linear)
                             for p in layer.parameters())

        print(f"Total parameters: {total_params_before:,}")
        print(f"Conv parameters: {conv_params_before:,} ({conv_params_before/total_params_before*100:.1f}%)")
        print(f"FC parameters: {fc_params_before:,} ({fc_params_before/total_params_before*100:.1f}%)")

        model_size_before = self.get_model_size(model)
        print(f"Model size: {model_size_before:.2f} MB")

        # Get total filters
        total_filters = sum(
            layer.out_channels
            for layer in model.net_1
            if isinstance(layer, torch.nn.Conv2d)
        )
        filters_to_prune = int(total_filters * pruning_amount)

        print(f"\nTotal filters: {total_filters}, Filters to prune: {filters_to_prune}")

        # Generate pruning targets
        pruner = FilterPruner(model)
        pruner.reset()

        # Generate dummy input
        dummy_input = torch.randn(16, 3, 224, 224).to(self.device)
        model.eval()
        # Forward pass to compute filter ranks
        _ = pruner.forward(dummy_input)

        pruner.normalize_ranks_per_layer()
        prune_targets = pruner.get_pruning_plan(filters_to_prune)

        # Track layers to be pruned
        layer_pruning_counts = {}
        for layer_index, _ in prune_targets:
            layer_pruning_counts[layer_index] = layer_pruning_counts.get(layer_index, 0) + 1

        print("\nPruning distribution by layer:")
        for layer_idx in sorted(layer_pruning_counts.keys()):
            layer_filter_count = 0
            for i, layer in enumerate(model.net_1):
                if i == layer_idx and isinstance(layer, torch.nn.Conv2d):
                    layer_filter_count = layer.out_channels

            if layer_filter_count > 0:
                prune_percent = (layer_pruning_counts[layer_idx] / layer_filter_count) * 100
                print(f"  Layer {layer_idx}: {layer_pruning_counts[layer_idx]}/{layer_filter_count} filters ({prune_percent:.1f}%)")

        # Get layer information before pruning
        self.get_layer_info(model)

        # Perform pruning
        pruning = Pruning(model)

        channels_per_layer = {
            i: layer.out_channels
            for i, layer in enumerate(model.modules())
            if isinstance(layer, torch.nn.Conv2d)
        }
        # Filter pruning targets to avoid pruning layers with only 1 channel
        filtered_prune_targets = []
        for layer_index, filter_index in prune_targets:
            if layer_index not in channels_per_layer:
                continue
            if channels_per_layer[layer_index] <= 1:
                print(f"Layer {layer_index} has only 1 channel, skipping pruning")
                continue

            channels_per_layer[layer_index] -= 1
            filtered_prune_targets.append((layer_index, filter_index))

        print(f"Actual filters to prune after filtering: {len(filtered_prune_targets)}")

        # Prune filters one by one
        for idx, (layer_index, filter_index) in enumerate(filtered_prune_targets):
            try:
                before_size = self.get_model_size(model)
                model = pruning.prune_vgg_conv_layer(model, layer_index, filter_index)
                after_size = self.get_model_size(model)

                if idx % 10 == 0:
                    print(f"Pruned {idx+1}/{len(filtered_prune_targets)} filters. Size change: {before_size:.2f}MB → {after_size:.2f}MB")
                    self._clear_memory()

            except Exception as e:
                print(f"Error pruning layer {layer_index}, filter {filter_index}: {e}")

        # Analyze model after pruning
        print("\nModel AFTER pruning:")
        total_params_after = sum(p.numel() for p in model.parameters())
        conv_params_after = sum(p.numel() for name, layer in enumerate(model.net_1) 
                              if isinstance(layer, torch.nn.Conv2d)
                              for p in layer.parameters())
        fc_params_after = sum(p.numel() for name, layer in enumerate(model.net_2) 
                            if isinstance(layer, torch.nn.Linear)
                            for p in layer.parameters())

        print(f"Total parameters: {total_params_after:,}")
        print(f"Conv parameters: {conv_params_after:,} ({conv_params_after/total_params_after*100:.1f}%)")
        print(f"FC parameters: {fc_params_after:,} ({fc_params_after/total_params_after*100:.1f}%)")

        model_size_after = self.get_model_size(model)
        print(f"Model size: {model_size_after:.2f} MB")

        # Get layer information after pruning
        self.get_layer_info(model)

        # Print summary
        param_reduction = (total_params_before - total_params_after) / total_params_before * 100
        size_reduction = (model_size_before - model_size_after) / model_size_before * 100

        print("\n------- Pruning Summary -------")
        print(f"Pruning amount: {pruning_amount*100:.1f}%")
        print(f"Original filters: {total_filters}")
        print(f"Filters pruned: {len(filtered_prune_targets)} ({len(filtered_prune_targets)/total_filters*100:.1f}%)")
        print(f"Parameters: {total_params_before:,} → {total_params_after:,} ({param_reduction:.1f}% reduction)")
        print(f"Model size: {model_size_before:.2f} MB → {model_size_after:.2f} MB ({size_reduction:.1f}% reduction)")

        return model

def main():
    parser = argparse.ArgumentParser(description="Test pruning effectiveness on model size")
    parser.add_argument("--prune", type=float, default=0.1, 
                        help="Pruning amount (0.1 = 10% of filters)")
    args = parser.parse_args()
    
    tester = PruningTester()
    model = tester.load_model()
    _ = tester.prune_model(model, args.prune)

if __name__ == "__main__":
    main()