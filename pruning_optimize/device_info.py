import psutil
import torch
import cpuinfo
import time
import numpy as np

class DeviceInfo:
    def __init__(self):
        self.memory_info = self.get_memory_info()
        self.gpu_info = self.get_gpu_flops()
        self.cpu_info = self.get_cpu_info()
    
    def get_cpu_l1_cache():
        return cpuinfo.get_cpu_info()['l1_data_cache_size']
    
    def get_cpu_flops(matrix_size=1000, iterations=100):
        device = 'cpu'
        A = torch.randn(matrix_size, matrix_size).to(device)
        B = torch.randn(matrix_size, matrix_size).to(device)
        
        start = time.time()
        for i in range(iterations):
            _ = np.dot(A, B)
        end = time.time()
        elapsed_time = end - start
        
        operations = 2*matrix_size**3*iterations
        flops = operations/elapsed_time
        return flops
    
    def total_memory():
        return psutil.virtual_memory().total
    
    def get_gpu_flops(matrix_size=1000, iterations=100):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        A = torch.randn(matrix_size, matrix_size).to(device)
        B = torch.randn(matrix_size, matrix_size).to(device)
    
        start = time.time()
        for i in range(iterations):
            _ = torch.matmul(A, B).to(device)
        end = time.time()
        elapsed_time = end - start
        
        operations = 2*matrix_size**3*iterations
        flops = operations/elapsed_time
        return flops