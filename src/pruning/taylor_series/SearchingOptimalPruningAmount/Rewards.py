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
        # Define a narrow acceptable range just above the minimum accuracy
        optimal_margin = 0.5  # Optimal range is min_acc to min_acc + 0.5%
        penalty_threshold = 2.0  # Start penalizing more aggressively if accuracy > min_acc + 2%
        
        if accuracy_new < self.min_acc:
            # Severe penalty for being below minimum - exponential penalty (keep as is)
            shortfall = self.min_acc - accuracy_new
            return -1500.0 * (1.0 + shortfall * 0.5) ** 2
        elif accuracy_new <= self.min_acc + optimal_margin:
            # Maximum reward for being just above minimum threshold (sweet spot)
            # The closer to min_acc, the higher the reward
            closeness = 1.0 - ((accuracy_new - self.min_acc) / optimal_margin)
            return 1200.0 * (0.8 + 0.2 * closeness)
        elif accuracy_new <= self.min_acc + penalty_threshold:
            # Linear decrease in reward as we move away from optimal margin
            excess = accuracy_new - (self.min_acc + optimal_margin)
            normalized_excess = excess / (penalty_threshold - optimal_margin)
            return 1000.0 * (1.0 - normalized_excess * 0.7)
        else:
            # Active penalty for accuracy much higher than needed - wastes capacity
            excess = accuracy_new - (self.min_acc + penalty_threshold)
            # Quadratically increasing penalty for excessive accuracy
            penalty_factor = min(1.0, (excess / 5.0) ** 1.5)
            base_reward = 300.0  # Still somewhat positive, but much lower
            return base_reward - (500.0 * penalty_factor)

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