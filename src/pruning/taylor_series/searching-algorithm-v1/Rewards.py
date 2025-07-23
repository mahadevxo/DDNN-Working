import numpy as np
import matplotlib.pyplot as plt

class Reward:
    def __init__(self, min_accuracy: float=80.0, max_model_size: float=350.0, x: float=1, y: float=1, z: float=1):
        self.min_accuracy = min_accuracy
        self.max_model_size = max_model_size
        self.x = x
        self.y = y
        self.z = z
        self.comp_scale_factor = 50
        self.size_scale_factor = 0.5

    def _get_accuracy_reward(self, curr_accuracy, sigma_right=4, sigma_left=2):
        diff = curr_accuracy - self.min_accuracy
        if 0<=diff<=1e-2:
            return (np.exp(- (diff**2) / (10 * sigma_left**2)) * 100)
        else:
            return 1*(np.exp(-(abs(diff)**1.5) / (2 * sigma_right**2)) * 100)

    def _get_comp_time_reward(self, current_comp_time, sigma=0.8):
        return np.exp(- (current_comp_time**2) / (2 * sigma**2))*self.comp_scale_factor

    def _get_model_size_reward(self, current_model_size, sigma_left=2):
        diff = current_model_size - self.max_model_size
        if current_model_size > self.max_model_size:
            return np.exp(- ((diff)**2) / (10 * sigma_left**2))*99*self.size_scale_factor
        if current_model_size == self.max_model_size:
            return 99*self.size_scale_factor
        else:
            return (99+(current_model_size/self.max_model_size))*self.size_scale_factor

    def _more_acc_less_size(self, accuracy, min_accuracy, size, max_model_size):
        if accuracy >= min_accuracy and size <= max_model_size:
            return ((accuracy-min_accuracy)*2) + (max_model_size-size)/2
        return 0

    def get_reward(self, p, accuracy=0, time=0, size=0) -> float:
        acc_reward = np.array(self._get_accuracy_reward(accuracy))
        time_reward = np.array(self._get_comp_time_reward(time))
        size_reward = np.array(self._get_model_size_reward(size))
        better_reward = self._more_acc_less_size(accuracy, self.min_accuracy, size, self.max_model_size)

        x, y, z = self.x/(self.x+self.y+self.z), self.y/(self.x+self.y+self.z), self.z/(self.x+self.y+self.z)

        return (x*acc_reward + y*time_reward + z*size_reward + better_reward + p/2)
    
    def plot_rewards(self):
        p = np.linspace(0, 1, 100)
        accuracy = np.linspace(0, 100, 100)
        time = np.linspace(0, 2, 100)
        size = np.linspace(0, 500, 100)
        
        rewards = [self.get_reward(p_val, accuracy=acc, time=t, size=s) for p_val, acc, t, s in zip(p, accuracy, time, size)]
        acc_rewards = [self._get_accuracy_reward(acc) for acc in accuracy]
        time_rewards = [self._get_comp_time_reward(t) for t in time]
        size_rewards = [self._get_model_size_reward(s) for s in size]
        
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 1)
        plt.plot(p, acc_rewards, label='Accuracy Reward', color='blue')
        plt.title('Accuracy Reward')
        plt.xlabel('Pruning Amount (p)')
        plt.ylabel('Reward')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(p, time_rewards, label='Computation Time Reward', color='orange')
        plt.title('Computation Time Reward')
        plt.xlabel('Pruning Amount (p)')
        plt.ylabel('Reward')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(p, size_rewards, label='Model Size Reward', color='green')
        plt.title('Model Size Reward')
        plt.xlabel('Pruning Amount (p)')
        plt.ylabel('Reward')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(14.5, 4))
        plt.plot(p, rewards, label='Total Reward', color='purple')
        plt.title('Total Reward vs p')
        plt.xlabel('p')
        plt.ylabel('Total Reward')
        plt.grid()
        plt.legend()
        plt.show()