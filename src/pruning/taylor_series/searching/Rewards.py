import numpy as np

class Reward:
    def __init__(self, min_acc, max_size, x=0.33, y=0.33, z=0.33, smooth_factor=0.1):
        """
        Initialize the reward function for model pruning optimization.
        
        Args:
            min_acc: Minimum acceptable accuracy (in percentage)
            max_size: Maximum acceptable model size
            x: Weight for accuracy reward component (default: 0.33)
            y: Weight for model size reward component (default: 0.33)
            z: Weight for computation time reward component (default: 0.33)
            smooth_factor: Smoothing factor for reward stability (default: 0.1)
        """
        self.min_acc = min_acc / 100
        self.max_size = max_size
        self.reward = 0
        self.prev_reward = 0
        self.smooth_factor = smooth_factor
        
        # Normalize weights to sum to exactly 1.0
        total = x + y + z
        self.x = x / total
        self.y = y / total
        self.z = z / total
        
        self.comp_time_last = None

    def _get_accuracy_reward(self, acc):
        """
        Calculate accuracy reward with a Gaussian peak and penalties for deviation.
        Strongly penalizes accuracy below minimum threshold.
        """
        # Hard constraint for minimum accuracy
        if acc < self.min_acc:
            # Exponential penalty for falling below minimum accuracy
            deficit = self.min_acc - acc
            return -10.0 * np.exp(20 * deficit)
        
        # Reward for being close to or slightly above minimum accuracy
        delta = acc - self.min_acc
        if delta <= 0.03:  # Very close to minimum - ideal zone
            return 3.0 - delta * 15  # Max reward at exactly min_acc
        elif delta <= 0.06:  # Slightly above minimum - good zone
            return 2.0 - (delta - 0.03) * 10
        else:  # Too much above minimum - wasting capacity
            return 1.5 - (delta - 0.06) * 5
    
    def _get_model_size_reward(self, model_size_new, size_width=1.0,
                              penalty_scale=5.0, penalty_sharpness=2.0,
                              param_reduction=None):
        """
        Calculate model size reward that strongly favors smaller models.
        Rewards decrease in size even significantly below maximum size.
        """
        # Hard constraint for maximum model size
        if model_size_new > self.max_size:
            # Strong exponential penalty for exceeding max size
            excess = model_size_new - self.max_size
            return -8.0 * np.exp(0.05 * excess)
        
        # Calculate how much smaller than max_size the model is (as a ratio)
        size_ratio = model_size_new / self.max_size
        
        # Reward function that increases as model gets smaller
        # Hyperbolic reward: approaches 5.0 as size approaches 0
        # 1.0 at max_size, 2.0 at 50% of max_size, 3.33 at 30% of max_size
        base_reward = 1.0 / size_ratio
        
        # Cap the reward to avoid extreme values
        capped_reward = min(5.0, base_reward)
        
        # Add bonus for very small models (less than 30% of max size)
        if size_ratio < 0.3:
            small_bonus = (0.3 - size_ratio) * 10
            return capped_reward + small_bonus
        
        return capped_reward

    def _get_comp_time_reward(self, comp_time_new, comp_time_last,
                              left_width=0.2, right_width=0.05, central_scale=1.0):
        """
        Calculate computation time reward that favors speed improvements.
        Returns a value between 0.0 and 1.0.
        """
        # Avoid division by zero
        comp_time_last = max(comp_time_last, 1e-6)
        
        # Calculate improvement ratio
        improvement_ratio = (comp_time_last - comp_time_new) / comp_time_last
        
        # Higher reward for faster models (lower computation time)
        if improvement_ratio >= 0:  # Improved or same speed
            return 1.0 * np.tanh(3 * improvement_ratio) + 0.5
        else:  # Slower than before
            return -2.0 * np.tanh(-3 * improvement_ratio)  # Penalty for slower models

    def getReward(self, accuracy, model_size, comp_time, comp_time_last=None, param_reduction=None):
        """
        Calculate overall reward for a pruned model combining accuracy, 
        model size and computation time rewards.
        
        Args:
            accuracy: Model accuracy after pruning (0.0-1.0)
            model_size: Size of the pruned model
            comp_time: Computation time of the pruned model
            comp_time_last: Previous computation time (if None, uses stored value)
            param_reduction: Parameter reduction ratio (optional)
            
        Returns:
            Normalized reward value suitable for optimization
        """
        if comp_time_last is not None:
            self.comp_time_last = comp_time_last
        elif self.comp_time_last is None:
            self.comp_time_last = comp_time
            
        # Print input values for debugging
        print(f"Evaluating - Accuracy: {accuracy:.2f}, Model Size: {model_size:.2f} ({model_size/self.max_size:.1%} of max), Comp Time: {comp_time:.4f}")

        # Calculate individual reward components
        acc_r = self._get_accuracy_reward(acc=accuracy)
        size_r = self._get_model_size_reward(model_size_new=model_size, param_reduction=param_reduction)
        time_r = self._get_comp_time_reward(comp_time_new=comp_time, comp_time_last=self.comp_time_last)
        
        # Print component rewards for debugging
        print(f"Component Rewards - Accuracy: {acc_r:.2f}, Size: {size_r:.2f}, Time: {time_r:.2f}")

        # Store current computation time for next iteration
        self.comp_time_last = comp_time

        # Hard constraints: heavily penalize violations
        if accuracy < self.min_acc:
            print(f"CONSTRAINT VIOLATION: Accuracy {accuracy:.2f} below minimum {self.min_acc*100:.2f}")
            # Already penalized in _get_accuracy_reward
            
        if model_size > self.max_size:
            print(f"CONSTRAINT VIOLATION: Model size {model_size:.2f} exceeds maximum {self.max_size:.2f}")
            # Already penalized in _get_model_size_reward
        
        # Add special bonus for achieving both small size AND meeting accuracy requirements
        if accuracy >= self.min_acc and model_size < self.max_size * 0.8:
            size_ratio = model_size / self.max_size
            combo_bonus = (1.0 - size_ratio) * 5.0
            print(f"Adding small-but-accurate bonus: +{combo_bonus:.2f}")
            acc_r += combo_bonus * 0.5  # Apply half to accuracy
            size_r += combo_bonus * 0.5  # Apply half to size

        # Calculate weighted sum of rewards
        raw_reward = (self.x * acc_r) + (self.y * size_r) + (self.z * time_r)
        
        # Clip reward to prevent extreme values
        clipped_reward = np.clip(raw_reward, -10.0, 10.0)
        
        # Apply smoothing for stability
        self.reward = (1 - self.smooth_factor) * clipped_reward + self.smooth_factor * self.prev_reward
        self.prev_reward = self.reward
        
        print(f"Final Reward: {self.reward:.4f}")
        return self.reward