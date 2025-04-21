import torch
import torch.nn as nn

class NetworkAdapter(nn.Module):
    """
    Adapter module that ensures compatibility between net_1 and net_2 after pruning
    Handles dimension mismatches by applying appropriate transformations
    """
    def __init__(self, model, adapter_mode='zero_pad'):
        super(NetworkAdapter, self).__init__()
        # Store references to the original nets for proper attribute access
        self.net_1 = model.net_1  # This fixes the attribute access issue
        self.net_2 = model.net_2
        
        # Store as model_1 and model_2 for the forward pass
        self.model_1 = model.net_1
        self.model_2 = model.net_2
        self.device = next(model.parameters()).device
        
        # Detect the actual spatial dimensions and feature counts
        self.input_channels, self.spatial_size = self._detect_spatial_dims(model)
        
        # Determine net_2 input size
        self.net_2_input_size = self._get_net_2_input_size(model.net_2)
        
        # Calculate expected flattened feature size
        self.expected_features = self._get_expected_features()
        
        # Check if we need adaptation
        self.needs_adaptation = self.expected_features != self.net_2_input_size
        self.adapter_mode = adapter_mode
        
        if self.needs_adaptation:
            print("NetworkAdapter: Dimension mismatch detected:")
            print(f"  - net_1 output: {self.expected_features} features")
            print(f"  - net_2 expects: {self.net_2_input_size} features")
            print(f"  - Using adapter mode: {adapter_mode}")
            
            if adapter_mode == 'zero_pad':
                # Zero padding doesn't need parameters
                pass
            elif adapter_mode == 'projection':
                # Linear projection to match dimensions
                self.projection = nn.Linear(
                    self.expected_features, 
                    self.net_2_input_size
                ).to(self.device)
            elif adapter_mode == 'adaptive':
                # Adaptive pooling to handle spatial dimension changes
                self.adaptive_pool = nn.AdaptiveAvgPool2d(
                    (int((self.net_2_input_size / self.input_channels) ** 0.5),) * 2
                )
            else:
                raise ValueError(f"Unknown adapter mode: {adapter_mode}")
    
    def _detect_spatial_dims(self, model):
        """Detect the spatial dimensions of net_1 output by running a dummy forward pass"""
        last_conv_channels = next(
            (
                module.out_channels
                for module in reversed(list(model.net_1))
                if isinstance(module, nn.Conv2d)
            ),
            0,
        )
        # Run a dummy forward pass to get spatial dimensions
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)  # Assume standard ImageNet size
        with torch.no_grad():
            x = model.net_1(dummy_input)

        # Extract spatial dimensions
        _, _, h, w = x.shape
        return last_conv_channels, (h, w)
    
    def _get_net_2_input_size(self, net_2):
        """Get the input size expected by net_2"""
        for module in net_2:
            if isinstance(module, nn.Linear):
                return module.in_features
        raise ValueError("Could not find a Linear layer in net_2")
    
    def _get_expected_features(self):
        """Calculate the expected flattened feature size from net_1 output"""
        return self.input_channels * self.spatial_size[0] * self.spatial_size[1]
    
    def forward(self, x):
        # Pass through net_1
        x = self.model_1(x)

        # Apply adaptation if needed
        if self.needs_adaptation:
            if self.adapter_mode == 'zero_pad':
                # Flatten features
                batch_size = x.size(0)
                x_flat = x.view(batch_size, -1)

                # Zero padding or truncation
                if x_flat.size(1) < self.net_2_input_size:
                    padding = torch.zeros(
                        batch_size, self.net_2_input_size - x_flat.size(1),
                        device=self.device
                    )
                    x_flat = torch.cat([x_flat, padding], dim=1)
                else:
                    # Truncate if too large
                    x_flat = x_flat[:, :self.net_2_input_size]

            elif self.adapter_mode == 'projection':
                # Flatten and project
                x_flat = x.view(x.size(0), -1)
                x_flat = self.projection(x_flat)

            elif self.adapter_mode == 'adaptive':
                # Adaptive pooling to get the right spatial dimensions
                x = self.adaptive_pool(x)
                x_flat = x.view(x.size(0), -1)
        else:
            # No adaptation needed, just flatten
            x_flat = x.view(x.size(0), -1)

        return self.model_2(x_flat)

class AdaptedModel(nn.Module):
    """Wrapper model that includes the adapter"""
    def __init__(self, original_model, adapter_mode='zero_pad'):
        super(AdaptedModel, self).__init__()
        self.adapter = NetworkAdapter(original_model, adapter_mode)
        
        # Correctly expose net_1 and net_2 to support size calculation
        self.net_1 = original_model.net_1
        self.net_2 = original_model.net_2
    
    def forward(self, x):
        return self.adapter(x)
