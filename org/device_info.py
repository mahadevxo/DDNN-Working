import psutil
import pynvml
import cpuinfo

class DeviceInfo:
    
    def __init__(self):
        self.memory_info = self.get_memory_info()
        self.gpu_info = self.get_gpu_flops()
        self.cpu_info = self.get_cpu_info()
    
    def get_memory_info(self):
        # Get RAM information
        ram_info = psutil.virtual_memory()
        ram_total = ram_info.total / (1024 ** 3)  # GB
        ram_available = ram_info.available / (1024 ** 3)  # GB

        # Get GPU memory information
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_memory_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_info.append({
                "gpu_name": pynvml.nvmlDeviceGetName(handle).decode("utf-8"),
                "total_memory": mem_info.total / (1024 ** 3),  # GB
                "free_memory": mem_info.free / (1024 ** 3),  # GB
            })
        pynvml.nvmlShutdown()

        return {"ram_total": ram_total, "ram_available": ram_available, "gpu_memory_info": gpu_memory_info}



    def get_gpu_flops(self):
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_flops_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            cuda_cores = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            clock_speed = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)  # MHz
            flops = cuda_cores * clock_speed * 1e6 * 2  # FLOPS = cores * clock * 2 (FMAs)
            gpu_flops_info.append({
                "gpu_name": pynvml.nvmlDeviceGetName(handle).decode("utf-8"),
                "flops": flops / 1e12  # TFLOPS
            })
        pynvml.nvmlShutdown()
        return gpu_flops_info

    def get_cpu_info(self):
        cpu_info = cpuinfo.get_cpu_info()
        cores = psutil.cpu_count(logical=False)
        threads = psutil.cpu_count(logical=True)
        return {
            "cpu_name": cpu_info['brand_raw'],
            "cores": cores,
            "threads": threads,
            "frequency": cpu_info.get('hz_actual', [None])[0] / 1e6 if cpu_info.get('hz_actual') else None,  # GHz
        }