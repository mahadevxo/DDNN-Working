import numpy as np

class ModelStats:
    def __init__(self):
        pass
    def get_accuracy(self, p):
        #boltzman sigmoidal
        
        bottom = -0.01748
        top = 0.8302
        v50 = 88.02
        slope = -2.660
        denom = 1 + np.exp((v50-p)/slope)
        
        acc = bottom + (top/denom)

        return max(acc, 0)

    def get_size(self, p):
        coeffs = [496.7, -706.2, 276.9, 4.020]
        return sum(c * p**i for i, c in enumerate(coeffs))

    def get_time(self, p, device_perf):
        b0 = 0.06386
        b1 = -0.06386
        t = (b0 + b1 * p)*10
        return t - (device_perf*t)
    
class Rewards:
    def __init__(self):
        self.model_stats = ModelStats()

    def get_accuracy_reward(self, curr_accuracy, min_accuracy, sigma_right=4, sigma_left=2):
        diff = curr_accuracy - min_accuracy
        if 0<=diff<=1e-2:
            return (np.exp(- (diff**2) / (10 * sigma_left**2)) * 100)
        else:
            return 1*(np.exp(- (abs(diff)**1.5) / (2 * sigma_right**2)) * 100)
        
    def get_comp_time_reward(self, current_comp_time, sigma=0.8):
        return np.exp(- (current_comp_time**2) / (2 * sigma**2))*10

    def get_model_size_reward(self, current_model_size, max_model_size, sigma_left=2):
        diff = current_model_size - max_model_size
        if current_model_size > max_model_size:
            return np.exp(- ((diff)**2) / (10 * sigma_left**2))*99*0.5
        if current_model_size == max_model_size:
            return 99*(0.5)
        else:
            return (99+(current_model_size/max_model_size))*0.5
        
    def more_acc_less_size(self, accuracy, min_accuracy, size, max_model_size):
        if accuracy >= min_accuracy and size <= max_model_size:
            return ((accuracy-min_accuracy)*2) + (max_model_size-size)/2
        return 0

    def get_reward(self, p, min_accuracy=80.0, max_model_size=350.0, x=10, y=1, z=1) -> float:
        accuracy = self.model_stats.get_accuracy(p)
        time = self.model_stats.get_time(p, device_perf=0)
        size = self.model_stats.get_size(p)

        acc_reward = np.array(self.get_accuracy_reward(accuracy, min_accuracy))
        time_reward = np.array(self.get_comp_time_reward(time))
        size_reward = np.array(self.get_model_size_reward(size, max_model_size))
        better_reward = self.more_acc_less_size(accuracy, min_accuracy, size, max_model_size)
        
        x, y, z = x/(x+y+z), y/(x+y+z), z/(x+y+z)
        
        return np.float64(x*acc_reward + y*time_reward + z*size_reward + better_reward + 0)

