import torch
import gc

class Pruning:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
    
    def _clear_memory(self):
        """Helper method to clear memory efficiently"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _replace_layers(self, model, i, indices, layers):
        return layers[indices.index(i)] if i in indices else model[i]
    
    def _get_next_conv(self, model, layer_index):
        next_conv = None
        offset = 1
        modules = list(model.net_1)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                next_conv = candidate
                break
            offset += 1
        return next_conv
    
    def _get_next_conv_offset(self, model, layer_index):
        offset = 1
        modules = list(model.net_1)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                break
            offset += 1
        return offset
    
    def _create_new_conv(self, conv, in_channels=None, out_channels=None):
        in_channels = conv.in_channels if in_channels is None else in_channels
        out_channels = conv.out_channels - 1 if out_channels is None else out_channels
        return torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
        ).to(self.device)
        
    def _prune_conv_layer(self, conv, new_conv, filter_index):
        with torch.no_grad():
            old_weights = conv.weight.data
            new_weights = torch.cat((
                old_weights[:filter_index],
                old_weights[filter_index+1:]
            ), dim=0)
            # replace the weight tensor in one go
            new_conv.weight.data = new_weights.clone().to(self.device)
            if conv.bias is not None:
                old_bias = conv.bias.data
                new_bias = torch.cat((
                    old_bias[:filter_index],
                    old_bias[filter_index+1:]
                ), dim=0)
                new_conv.bias.data = new_bias.clone().to(self.device)
    
    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_index):
        with torch.no_grad():
            old_weights = next_conv.weight.data
            new_weights = torch.cat((
                old_weights[:, :filter_index],
                old_weights[:, filter_index+1:]
            ), dim=1)
            new_next_conv.weight.data = new_weights.clone().to(self.device)
            if next_conv.bias is not None:
                new_next_conv.bias.data = next_conv.bias.data.clone().to(self.device)
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index):
        with torch.no_grad():
            # Replace conv layer in model
            modules = list(model.net_1)
            modules[layer_index] = new_conv
            model.net_1 = torch.nn.Sequential(*modules)
            
            # Find the first fully connected layer
            layer_index = 0
            old_linear_layer = None
            for _, module in model.net_2._modules.items():
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    break
                layer_index += 1
            
            if old_linear_layer is None:
                raise ValueError(f"No linear layer found in classifier, Model: {model}")
            
            # Calculate parameters per input channel
            params_per_input_channel = old_linear_layer.in_features // conv.out_channels
            
            # Create new linear layer
            new_linear_layer = torch.nn.Linear(
                old_linear_layer.in_features - params_per_input_channel,
                old_linear_layer.out_features
            ).to(self.device)
            
            # Update the linear layer weights on GPU
            old_weights = old_linear_layer.weight.data
            left = old_weights[:, :filter_index * params_per_input_channel]
            right = old_weights[:, (filter_index + 1) * params_per_input_channel:]
            new_weights = torch.cat((left, right), dim=1)
            
            # Ensure proper memory layout
            new_weights = new_weights.contiguous()
            new_linear_layer.weight.data.copy_(new_weights)
            new_linear_layer.bias.data.copy_(old_linear_layer.bias.data)
            
            # Replace in the model's net_2 part
            modules = list(model.net_2)
            modules[layer_index] = new_linear_layer
            model.net_2 = torch.nn.Sequential(*modules)
        
        self._clear_memory()
        return model
    
    def prune_conv_layers(self, model, layer_index, filter_index):
        modules = list(model.net_1)
        conv_indices = [i for i, m in enumerate(modules) if isinstance(m, torch.nn.Conv2d)]
        # allow layer_index as conv‐layer count or as actual module index
        if layer_index in conv_indices:
            actual_idx = layer_index
        else:
            actual_idx = conv_indices[int(layer_index)]
        conv = modules[actual_idx]

        # guard against pruning the last remaining filter
        if conv.out_channels <= 1:
            # print(f"Skipping pruning on layer {layer_index}: only {conv.out_channels} filters remain")
            return model

        # Prune this conv
        next_conv = self._get_next_conv(model, actual_idx)
        new_conv = self._create_new_conv(conv)
        self._prune_conv_layer(conv, new_conv, filter_index)

        if next_conv is not None:
            # Find the module index of the next Conv2d
            next_idx = next(i for i, m in enumerate(modules) if m is next_conv)
            # Build and prune the next conv
            next_new_conv = self._create_new_conv(
                next_conv,
                in_channels=next_conv.in_channels - 1,
                out_channels=next_conv.out_channels
            )
            self._prune_next_conv_layer(next_conv, next_new_conv, filter_index)
            # Replace modules and rebuild Sequential
            modules[actual_idx] = new_conv
            modules[next_idx] = next_new_conv
            model.net_1 = torch.nn.Sequential(*modules)
        else:
            # Last conv: update classifier accordingly
            model = self._prune_last_conv_layer(model, conv, new_conv, actual_idx, filter_index)

        self._clear_memory()
        return model