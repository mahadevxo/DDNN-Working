class Reward:
    def __init__(self, min_acc, min_size, x=0.33, y=0.33, z=0.33):
        self.min_acc = min_acc
        self.min_size = min_size
        self.reward = 0
        self.x = x
        self.y = y
        self.z = z
        
        print(f"Reward initialized with min_acc: {self.min_acc}, min_size: {self.min_size}, x: {self.x}, y: {self.y}, z: {self.z}")
        
    def _get_accuracy_reward(self, accuracy_new):
        # More aggressive accuracy reward with steeper penalties and rewards
        threshold_margin = 1.0  # Use a 1% margin
        
        if abs(accuracy_new - self.min_acc) <= threshold_margin:
            # Reward for being close to the minimum accuracy
            # Higher reward when closer to the exact threshold
            closeness = 1.0 - (abs(accuracy_new - self.min_acc) / threshold_margin)
            return 800.0 * (0.5 + 0.5 * closeness)  # Increased base reward
        elif accuracy_new > (self.min_acc + threshold_margin):
            # Smaller reward for exceeding minimum by too much (we want to maximize pruning)
            excess = accuracy_new - self.min_acc - threshold_margin
            # Non-linear reward for excess accuracy
            return 400.0 + (excess * 25) * (1.0 + excess/10.0)
        else:
            # More severe penalty for being below minimum accuracy - exponential penalty
            shortfall = self.min_acc - accuracy_new
            return -1500.0 * (1.0 + shortfall * 0.5) ** 2

    def _get_model_size_reward(self, model_size_new, param_reduction=None):
        # More aggressive reward for parameter reduction
        if param_reduction is not None:
            # Exponential reward based on percentage of parameters reduced
            # Strongly favor higher pruning rates
            reduction_reward = 15.0 * (param_reduction ** 1.5)  # Non-linear scaling
            return min(reduction_reward, 1200)  # Higher cap
        
        # Fall back to the previous approach with more aggressive scaling
        delta_model_size = model_size_new - self.min_size
        if delta_model_size >= 0:
            # Steeper penalty for exceeding target size
            return -500 * delta_model_size * (1.0 + delta_model_size/10.0)
        
        # More aggressive reward for smaller models
        reduction_factor = abs(delta_model_size) / self.min_size
        # Exponential reward for high reduction
        return 600 * min(reduction_factor ** 1.3 * 2, 1.5) + 300

    def _get_comp_time_reward(self, comp_time_new, comp_time_last):
        # More aggressive reward for computation time improvements
        if comp_time_last > 0:
            relative_change = (comp_time_last - comp_time_new) / comp_time_last
            
            if relative_change > 0:  # Speedup
                # Reward speedups more aggressively
                return 350 * (relative_change ** 1.2)  # Non-linear reward for speedups
            else:  # Slowdown
                # Penalize slowdowns more severely
                return 250 * relative_change  # Linear penalty for slowdowns
        return 0

    def getReward(self, accuracy, model_size, comp_time, comp_time_last, param_reduction=None):
        if not (0.99 <= self.x + self.y + self.z <= 1.01):
            raise ValueError("x, y, and z must sum to 1 or 0")
        
        accuracy_reward = self._get_accuracy_reward(accuracy)
        model_size_reward = self._get_model_size_reward(model_size, param_reduction)
        comp_time_reward = self._get_comp_time_reward(comp_time, comp_time_last)
        
        # For debugging
        print(f"Rewards - Accuracy: {accuracy_reward:.2f}, Size/Param: {model_size_reward:.2f}, Time: {comp_time_reward:.2f}")
        
        # Apply a scaling factor to amplify differences in total reward
        raw_reward = (self.x*accuracy_reward) + (self.y*model_size_reward) + (self.z*comp_time_reward)
        
        # Apply sigmoid-like scaling to maintain values in reasonable range while amplifying differences
        if raw_reward > 0:
            reward = raw_reward * (1.0 + min(raw_reward/500.0, 1.0))
        else:
            reward = raw_reward * (1.0 + min(abs(raw_reward)/500.0, 1.0))
            
        return reward, comp_time