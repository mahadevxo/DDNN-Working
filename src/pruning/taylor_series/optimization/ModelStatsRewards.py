import numpy as np
import pickle

class ModelStats:
    def __init__(self):
        self.xgbmodel = pickle.load(
            open("../pruning-mvcnn-accuracy/xgboost_model.pkl", "rb")
        )
    def get_model_size(self, p): # for p as a vector
        # Treat individual pruning values correctly
        if np.isscalar(p) or len(p) == 1:
            p_val = p if np.isscalar(p) else p[0] # type: ignore
            coeffs = [507.9, -8.516, 0.04994, -7.665*1e-005]
            return sum(c * (p_val*100)**i for i, c in enumerate(coeffs)) # type: ignore
        else:
            # Handle vector case - calculate size for each element
            return np.array([self.get_model_size(p_i) for p_i in p])

    def get_inf_time(self, p, device_perf):
        base_time = 100 
        pruning_factor = np.exp(-2 * p)
        device_factor = 1 + 2 * (1 - device_perf)
        
        return base_time * pruning_factor * device_factor

    def get_model_accuracy(self, p):
        accuracy = self.xgbmodel.predict([p])
        for i in range(len(accuracy)):
            if accuracy[i] < 0:
                accuracy[i] = 0
        return accuracy

class Rewards:
    def __init__(self):
        self.model_stats = ModelStats()
        
    def _get_accuracy_reward(self, curr_accuracy, min_accuracy, sigma_right=4, sigma_left=2): # for p as a vector
        diff = curr_accuracy - min_accuracy
        if 0 <= diff <= 1e-2:
            return (np.exp(- (diff**2) / (10 * sigma_left**2)) * 100)
        else:
            return 1 * (np.exp(- (abs(diff)**1.5) / (2 * sigma_right**2)) * 100)
    
    def _get_comp_time_reward(self, current_comp_time, sigma=0.8): # for each model
        return np.exp(- (current_comp_time**2) / (2 * sigma**2)) * 10
    
    def _get_model_size_reward(self, current_model_size, max_model_size, sigma_left=2): # for each model
        diff = current_model_size - max_model_size
        if current_model_size > max_model_size:
            return np.exp(- ((diff)**2) / (10 * sigma_left**2)) * 99 * 0.5
        if current_model_size == max_model_size:
            return 99 * (0.5)
        else:
            return (99 + (current_model_size / max_model_size)) * 0.5
    
    def more_acc_less_size(self, accuracy, min_accuracy, size, max_model_size):
        if accuracy >= min_accuracy and size <= max_model_size:
            return ((accuracy - min_accuracy) * 2) + (max_model_size - size) / 2
        return 0
    
    def get_reward(self, p, min_accuracy, max_model_size, x=10, y=1, z=5):
        accuracy = self.model_stats.get_model_accuracy(p)
        times = [self.model_stats.get_inf_time(pv, device_perf=0) for pv in p]
        sizes = [self.model_stats.get_model_size(pv) for pv in p]

        acc_reward = np.array(self._get_accuracy_reward(accuracy, min_accuracy))
        
        size_reward = 0.0
        better_reward = 0.0
        time_reward = 0.0
        
        for i in range(len(sizes)):
            size_reward += self._get_model_size_reward(sizes[i], max_model_size[i])
            time_reward += self._get_comp_time_reward(times[i])
        
        num_violators = sum(1 for size in sizes if size > max_model_size[0])
        if num_violators > 0:
            better_reward=0
        else:
            better_reward += sum(self.more_acc_less_size(accuracy, min_accuracy, sizes[i], max_model_size[i]) for i in range(len(sizes)))

        return np.float64(x*acc_reward + y*time_reward + z*size_reward + better_reward + 0)