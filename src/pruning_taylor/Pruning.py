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
        
        # Safety check to prevent zero-sized dimensions
        in_channels = max(1, in_channels)
        out_channels = max(1, out_channels)
        
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
        # sourcery skip: low-code-quality
        # Create a copy of the model to update the features first
        model_copy = copy.deepcopy(model)

        # Update the features part with our new conv layer
        modules = list(model_copy.features)
        modules[layer_index] = new_conv
        model_copy.features = torch.nn.Sequential(*modules)

        # Calculate old and new flattened feature map sizes
        try:
            old_features_size = self._calculate_feature_map_size(model)
        except Exception:
            # If original model fails, use adaptive pooling to ensure compatibility
            old_features_size = 25088  # Default for VGG16

        try:
            new_features_size = self._calculate_feature_map_size(model_copy)
        except Exception:
            # If calculation fails, use adaptive pooling to ensure compatibility
            new_features_size = None

        # First, update the model's features
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

        # If we couldn't calculate the feature size, or if they don't match,
        # add adaptive pooling to ensure output size matches what classifier expects
        if new_features_size is None or new_features_size != old_linear_layer.in_features:
            # Add adaptive pooling to ensure the output size is correct
            # Calculate the spatial dimensions needed for the target size
            target_in_features = old_linear_layer.in_features

            # For VGG-like architectures, add pooling between features and classifier
            new_features = list(model.features)

            # Get the number of channels from the last conv layer
            last_conv = None
            for layer in reversed(new_features):
                if isinstance(layer, torch.nn.Conv2d):
                    last_conv = layer
                    break

            if last_conv:
                out_channels = last_conv.out_channels
                # Calculate spatial dimensions needed for the target input size
                spatial_dim = int(np.sqrt(target_in_features / out_channels))

                # Add adaptive pooling as the last layer in features
                adaptive_pool = torch.nn.AdaptiveAvgPool2d((spatial_dim, spatial_dim))

                # Replace or add adaptive pooling
                pool_added = False
                for i, layer in enumerate(new_features):
                    if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
                        new_features[i] = adaptive_pool
                        pool_added = True
                        break

                if not pool_added:
                    new_features.append(adaptive_pool)

                model.features = torch.nn.Sequential(*new_features)

            # No need to modify classifier if we adapt features to match its input size
        else:
            # If the new feature size is different, create and update the first linear layer
            new_linear_layer = torch.nn.Linear(
                new_features_size,
                old_linear_layer.out_features
            )

            # Initialize new linear layer with zeros or interpolated weights
            if new_features_size > old_linear_layer.in_features:
                new_weights = torch.zeros(old_linear_layer.out_features, new_features_size, device=self.device)
                new_weights[:, :old_linear_layer.in_features] = old_linear_layer.weight.data
            else:
                # For smaller feature maps, try simple scaling
                scale_factor = new_features_size / old_linear_layer.in_features
                # Use simple slicing for scaling down
                new_weights = old_linear_layer.weight.data[:, :new_features_size]

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
        
        # Safety check - don't prune if we're already at minimum size
        if conv.out_channels <= 1:
            print(f"WARNING: Cannot prune layer {layer_index} with only {conv.out_channels} output channels")
            return model
            
        next_conv = self._get_next_conv(model, layer_index)
        new_conv = self._create_new_conv(conv)
        
        self._prune_conv_layer(conv, new_conv, filter_index)
        
        if next_conv is not None:
            # Safety check for next conv layer's input channel count
            if next_conv.in_channels <= 1:
                print(f"WARNING: Cannot reduce input channels for next conv which already has {next_conv.in_channels} input channels")
                return model
                
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