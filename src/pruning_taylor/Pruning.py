import torch
import numpy as np

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
        
    def _prune_conv_layer(self, conv, new_conv, filter_index, device = 'mps'):
        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()
        
        new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index+1:, :, :, :]
        
        new_conv.weight.data = torch.from_numpy(new_weights).to(device)
        bias_numpy = conv.bias.data.cpu().numpy()
        
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index:] = bias_numpy[filter_index+1:]
        
        new_conv.bias.data = torch.from_numpy(bias).to(device)
    
    def _prune_next_conv_layer(self, next_conv, new_next_conv, filter_index, device = 'mps'):
        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = new_next_conv.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index+1:, :, :]
        
        new_next_conv.weight.data = torch.from_numpy(new_weights).to(device)
        new_next_conv.bias.data = next_conv.bias.data.to(device)
    
    def _prune_last_conv_layer(self, model, conv, new_conv, layer_index, filter_index, device = 'mps'):
        model.features = torch.nn.Sequential(
            *(self.replace_layers(model.features, i, [layer_index], \
                [new_conv]) for i, _ in enumerate(model.features)))
        
        layer_index = 0
        old_linear_layer = None
        for _, module in model.classifier._modules.items():
            # Fix: Check if module is a Linear layer, not the model itself
            if isinstance(module, torch.nn.Linear):
                old_linear_layer = module
                break
            layer_index += 1
        
        if old_linear_layer is None:
            raise ValueError(f"No linear layer found in classifier, Model: {model}")
        params_per_input_channel = old_linear_layer.in_features // conv.out_channels
        
        new_linear_layer = torch.nn.Linear(
            old_linear_layer.in_features - params_per_input_channel,
            old_linear_layer.out_features
        )
        
        old_weights = old_linear_layer.weight.data.cpu().numpy()
        new_weights = new_linear_layer.weight.data.cpu().numpy()
        
        new_weights[:, :filter_index * params_per_input_channel] = \
            old_weights[:, :filter_index * params_per_input_channel]
        new_weights[:, filter_index * params_per_input_channel:] = \
            old_weights[:, (filter_index + 1) * params_per_input_channel:]
            
        new_linear_layer.weight.data = torch.from_numpy(new_weights).to(device)
        new_linear_layer.bias.data = old_linear_layer.bias.data.to(device)
        
        model.classifier = torch.nn.Sequential(
            *(self._replace_layers(model.classifier, i, [layer_index], \
                [new_linear_layer]) for i, _ in enumerate(model.classifier)))
        
        return model
    
    def prune_vgg_conv_layer(self, model, layer_index, filter_index, device='mps'):
        _, conv = list(model.features._modules.items())[layer_index]
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
            model = self._prune_last_conv_layer(model, conv, new_conv, layer_index, filter_index, device)
        return model