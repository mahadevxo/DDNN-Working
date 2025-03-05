import torch
import numpy as np
import copy

class Pruning:
    def __init__(self, model):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
    
    def _replace_layers(self, model, i, indices, layers):
        return layers[indices.index(i)] if i in indices else model[i]
    
    def _get_next_conv(self, model, layer_index):
        next_conv = None
        offset = 1
        modules = list(model.features)
        while (layer_index + offset) < len(modules):
            candidate = modules[layer_index + offset]
            if isinstance(candidate, torch.nn.Conv2d):
                next_conv = candidate
                break
            offset += 1
        return next_conv
    
    def _get_next_conv_offset(self, model, layer_index):
        offset = 1
        modules = list(model.features)
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
        )
        
    def _prune_conv_layer(self, conv, new_conv, filter_index):
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        
        new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index+1:, :, :, :]
        
        new_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        bias_numpy = conv.bias.data.cpu().numpy()
        
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index+1:]
        
        new_conv.bias.data = torch.from_numpy(bias).to(self.device)
    
    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_index):
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index+1:, :, :]
        
        new_next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
        new_next_conv.bias.data = next_conv.bias.data.to(self.device)
    
    def _calculate_feature_map_size(self, model, input_size=(224, 224)):
        """Calculate the size of flattened feature maps after convolutional layers"""
        # Create a dummy input to trace the size through the network
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1]).to(self.device)
        with torch.no_grad():
            # Pass through features only (convolutional part)
            x = model.features(dummy_input)
            # Return the flattened size
            return x.view(x.size(0), -1).shape[1]
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index):
        # Create a copy of the model to update the features first
        model_copy = copy.deepcopy(model)
        
        # Update the features part with our new conv layer
        modules = list(model_copy.features)
        modules[layer_index] = new_conv
        model_copy.features = torch.nn.Sequential(*modules)
        
        # Calculate old and new flattened feature map sizes
        old_features_size = self._calculate_feature_map_size(model)
        new_features_size = self._calculate_feature_map_size(model_copy)
        
        # Now update the original model's features
        modules = list(model.features)
        modules[layer_index] = new_conv
        model.features = torch.nn.Sequential(*modules)
        
        # Find the first linear layer in classifier
        layer_index = 0
        old_linear_layer = None
        for name, module in model.classifier._modules.items():
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index += 1
        
        if old_linear_layer is None:
            raise ValueError(f"No linear layer found in classifier, Model: {model}")
        
        # Create a new linear layer with adjusted input size
        new_linear_layer = torch.nn.Linear(
            new_features_size,
            old_linear_layer.out_features
        )
        
        # Transfer weights intelligently based on the ratio of feature map reduction
        old_weights = old_linear_layer.weight.data
        reduction_ratio = new_features_size / old_features_size
        
        # For assigning weights, use a simplified proportional assignment
        # This is an approximation but should maintain some of the trained information
        if reduction_ratio < 1.0:  # Only if we're reducing the size
            # Interpolate weights to the new size
            new_weights = torch.nn.functional.interpolate(
                old_weights.unsqueeze(0).unsqueeze(0),
                size=(new_features_size, old_linear_layer.out_features),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            new_weights = new_weights.t()  # Transpose to match linear layer format
        else:
            # If not reducing, initialize with zeros and copy what we can
            new_weights = torch.zeros(old_linear_layer.out_features, new_features_size, device=self.device)
            new_weights[:, :old_features_size] = old_weights
            
        new_linear_layer.weight.data = new_weights
        new_linear_layer.bias.data = old_linear_layer.bias.data.clone()
        
        # Replace the linear layer in the classifier
        classifier_modules = list(model.classifier)
        classifier_modules[layer_index] = new_linear_layer
        model.classifier = torch.nn.Sequential(*classifier_modules)
        
        return model
    
    def prune_vgg_conv_layer(self, model, layer_index, filter_index):
        modules = list(model.features._modules.items())
        if layer_index >= len(modules):
            raise ValueError(f"Layer index {layer_index} out of range, max: {len(modules)-1}")
            
        _, conv = modules[layer_index]
        if not isinstance(conv, torch.nn.Conv2d):
            raise ValueError(f"Layer at index {layer_index} is not a Conv2d layer")
            
        next_conv = self._get_next_conv(model, layer_index)
        new_conv = self._create_new_conv(conv)
        
        self._prune_conv_layer(conv, new_conv, filter_index)
        
        if next_conv is not None:
            # Fix: explicitly pass out_channels to keep the same number of filters
            next_new_conv = self._create_new_conv(next_conv, in_channels=next_conv.in_channels - 1, out_channels=next_conv.out_channels)
            self._prune_next_conv_layer(next_conv, next_new_conv, filter_index)
            # Replace specific layers in the Sequential rather than building tuples
            modules = list(model.features)
            modules[layer_index] = new_conv
            offset = self._get_next_conv_offset(model, layer_index)
            modules[layer_index + offset] = next_new_conv
            model.features = torch.nn.Sequential(*modules)
        else:
            # Use _prune_last_conv_layer to update classifier layers for the last conv layer
            model = self._prune_last_conv_layer(model, conv, new_conv, layer_index, filter_index)
        return model