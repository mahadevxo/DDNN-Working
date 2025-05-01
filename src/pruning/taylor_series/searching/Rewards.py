import numpy as np
class Reward:
    def __init__(self, min_acc, max_size, x=0.33, y=0.33, z=0.33):
        self.min_acc = min_acc/100 #min accuracy is in %, convert it to 0-1
        self.max_size = max_size #in MB
        self.reward = 0
        self.x = x
        self.y = y
        self.z = z
        
        print(f"Reward initialized with min_acc: {self.min_acc}, max_size: {self.max_size}, x: {self.x}, y: {self.y}, z: {self.z}")
        
    def _get_accuracy_reward(self, acc, accuracy_range=0.1):
        delta = acc - self.min_acc

        central = np.exp(-((delta / accuracy_range) ** 2)) * 1e4

        left_penalty = -1e3 * (1 / (1 + np.exp(50 * (acc - (self.min_acc - accuracy_range))))) ** 1e2

        right_penalty = -1e3 * (1 / (2 + np.exp(-50 * (acc - (self.min_acc + accuracy_range)))))

        return (central + left_penalty + right_penalty)

    def _get_model_size_reward(self, model_size_new, size_width=5, central_scale=1e4, central_width_scale=0.4, penalty_scale=1e4, penalty_sharpness=1, param_reduction=None):
    
        model_size_new /= 100
        self.max_size /= 100
        delta = model_size_new - self.max_size

        central = np.exp(-((delta / (size_width * central_width_scale)) ** 2)) * central_scale
        left_penalty = -penalty_scale * (1 / (1 + np.exp(penalty_sharpness * (model_size_new - (self.max_size - size_width)))))
        right_penalty = -penalty_scale * (1 / (1 + np.exp(-penalty_sharpness * (model_size_new - (self.max_size + size_width)))))

        return (central + left_penalty + right_penalty)

    def _get_comp_time_reward(self, comp_time_new, comp_time_last, left_width=0.1, right_width=0.1, central_scale=1e4):
        delta = (comp_time_new - comp_time_last) / comp_time_last
        if delta <= 0:
            reward = np.exp(-((delta / left_width) ** 2)) * central_scale
        else:
            reward = np.exp(-((delta / right_width) ** 2)) * central_scale

        return reward

    def getReward(self, accuracy, model_size, comp_time, comp_time_last, param_reduction=None):
        if not (0.99 <= self.x + self.y + self.z <= 1.01):
            raise ValueError("x, y, and z must sum to 1 or 0")
        
        accuracy_reward = self._get_accuracy_reward(accuracy)
        model_size_reward = self._get_model_size_reward(model_size, param_reduction)
        comp_time_reward = self._get_comp_time_reward(comp_time, comp_time_last)
        
        # For debugging
        print(f"Rewards - Accuracy: {accuracy_reward:.2f}, Size/Param: {model_size_reward:.2f}, Time: {comp_time_reward:.2f}")
        
        # Apply a scaling factor with more emphasis on accuracy when negative
        raw_reward = (self.x*accuracy_reward) + (self.y*model_size_reward) + (self.z*comp_time_reward)
        
        # Modified scaling to prioritize accuracy constraints
        if accuracy < self.min_acc:
            # When accuracy is below minimum, let that dominate the reward
            return raw_reward * 1.5  # Apply extra penalty multiplier
        elif raw_reward > 0:
            reward = raw_reward * (1.0 + min(raw_reward/600.0, 0.8))  # Reduced amplification
        else:
            reward = raw_reward * (1.0 + min(abs(raw_reward)/600.0, 0.8))  # Reduced amplification
            
        return reward