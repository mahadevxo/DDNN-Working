import torch
import numpy as np
import cv2
from device_info import DeviceInfo

def apply_pruning(model, sparsity_ratio):
    for _, module in model.nameed_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weights = module.weight.data.abs().view(-1)
            threshold = torch.quantile(weights, sparsity_ratio)

            mask = module.weight.data.abs() > threshold
            module.weight.data.mul_(mask)

    return model

def compress_feature_map(feature_map, dct_blocks, quality = 0.5):
    feature_map = feature_map.cpu().numpy()
    norm_feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
    compressed_data = []
    
    for batch in norm_feature_map:
        for channel in batch:
            h, w = channel.shape
            channel_dct = np.zeros_like(channel, np.float32)
            for i in range(0, h, dct_blocks):
                for j in range(0, w, dct_blocks):
                    channel_dct[i:i+dct_blocks, j:j+dct_blocks] = cv2.dct(channel[i:i+dct_blocks, j:j+dct_blocks])
                    
            quantized = (channel_dct/quality).astype(np.int16)
            
            compressed_data.append(quantized)
            
def optimization_model(device_params):
    #find computing power of the device
    memory_info = DeviceInfo().memory_info
    gpu_info = DeviceInfo().gpu_info
    cpu_info = DeviceInfo().cpu_info
    
    
    

def optimize_model(model, accuracy_threshold):
    
    pass