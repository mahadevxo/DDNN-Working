import torch
import numpy as np
class ComprehensiveVGGPruner:
    def __init__(self, model, prune_percentage=0.5):
        self.model = model
        self.prune_percentage = prune_percentage
        self.conv_layers = self.get_conv_layer_indices()
        
    def get_conv_layer_indices(self):
        # sourcery skip: inline-immediately-returned-variable, list-comprehension
        conv_indices = []
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, torch.nn.Conv2d):
                conv_indices.append(i)
        return conv_indices
    
    def calculate_filters_per_layer(self):
        filters_to_prune = {}
        for layer_idx in self.conv_layers:
            layer = list(self.model.features._modules.items())[layer_idx][1]
            n_filters = layer.out_channels
            n_to_prune = int(n_filters * self.prune_percentage)
            # Ensure we keep at least 2 filters per layer
            n_to_prune = min(n_to_prune, n_filters - 2)
            filters_to_prune[layer_idx] = n_to_prune
        return filters_to_prune
    
    def prune_conv_layer(self, model, layer_index, filter_index):
        _, conv = list(model.features._modules.items())[layer_index]
        next_conv = None
        offset = 1

        # Find the next Conv2D layer if it exists
        while layer_index + offset < len(model.features._modules.items()):
            res = list(model.features._modules.items())[layer_index + offset]
            if isinstance(res[1], torch.nn.Conv2d):
                _, next_conv = res
                break
            offset += 1

        # Create a new Conv2D layer with one fewer filter
        new_conv = torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels - 1,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        )

        # Copy old weights and prune one filter
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()

        # Copy weights except for the pruned filter
        new_weights[:filter_index] = old_weights[:filter_index]
        new_weights[filter_index:] = old_weights[filter_index + 1:]
        new_conv.weight.data = torch.from_numpy(new_weights).to('mps')

        # Handle biases
        if conv.bias is not None:
            bias_numpy = conv.bias.data.cpu().numpy()
            bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
            bias[:filter_index] = bias_numpy[:filter_index]
            bias[filter_index:] = bias_numpy[filter_index + 1:]
            new_conv.bias.data = torch.from_numpy(bias).to('mps')

        # Handle the next Conv2D layer if it exists
        if next_conv is not None:
            next_new_conv = torch.nn.Conv2d(
                in_channels=next_conv.in_channels - 1,
                out_channels=next_conv.out_channels,
                kernel_size=next_conv.kernel_size,
                stride=next_conv.stride,
                padding=next_conv.padding,
                dilation=next_conv.dilation,
                groups=next_conv.groups,
                bias=(next_conv.bias is not None),
            )

            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            # Copy weights excluding the pruned filter
            new_weights[:, :filter_index] = old_weights[:, :filter_index]
            new_weights[:, filter_index:] = old_weights[:, filter_index + 1:]

            next_new_conv.weight.data = torch.from_numpy(new_weights).to('mps')
            next_new_conv.bias.data = next_conv.bias.data if next_conv.bias is not None else None

            # Replace both layers
            model.features[layer_index] = new_conv
            model.features[layer_index + offset] = next_new_conv
        else:
            # Replace only the current layer
            model.features[layer_index] = new_conv

            # Handle the first linear layer in the classifier
            layer_index = 0
            old_linear_layer = None

            for i, module in enumerate(model.classifier):
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    layer_index = i
                    break

            if old_linear_layer is None:
                raise ValueError("No linear layer found in classifier")

            params_per_input_channel = old_linear_layer.in_features // conv.out_channels

            # Create a new Linear layer with fewer input features
            new_linear_layer = torch.nn.Linear(
                old_linear_layer.in_features - params_per_input_channel,
                old_linear_layer.out_features,
            )

            # Copy weights excluding the pruned parameters
            old_weights = old_linear_layer.weight.data.cpu().numpy()
            new_weights = new_linear_layer.weight.data.cpu().numpy()

            new_weights[:, :filter_index * params_per_input_channel] = \
                old_weights[:, :filter_index * params_per_input_channel]

            new_weights[:, filter_index * params_per_input_channel:] = \
                old_weights[:, (filter_index + 1) * params_per_input_channel:]

            new_linear_layer.weight.data = torch.from_numpy(new_weights).to('mps')
            new_linear_layer.bias.data = old_linear_layer.bias.data

            # Replace the linear layer
            model.classifier[layer_index] = new_linear_layer

        return model
    
    def prune_all_layers(self):
        # print("Starting comprehensive pruning of VGG16...")

        # Calculate number of filters to remove per layer
        filters_to_prune = self.calculate_filters_per_layer()

        # Prune each layer from last to first
        # We go backwards because earlier layer changes affect later layers
        for layer_idx in reversed(self.conv_layers):
            n_filters = filters_to_prune[layer_idx]
            # print(f"\nPruning layer {layer_idx}")
            # print(f"Original filters: {self.model.features[layer_idx].out_channels}")
            # print(f"Removing {n_filters} filters")

            # Remove filters one by one
            for _ in range(n_filters):
                # Always remove the first filter as the indices shift after each removal
                self.model = self.prune_conv_layer(self.model, layer_idx, 0)

            # print(f"Remaining filters: {self.model.features[layer_idx].out_channels}")

        print("Pruning completed!")
        return self.model