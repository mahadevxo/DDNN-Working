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
        # Increase the margin around min_acc to be considered "at threshold"
        threshold_margin = 1.0  # Use a 1% margin
        
        if abs(accuracy_new - self.min_acc) <= threshold_margin:
            # Reward for being close to the minimum accuracy
            # Higher reward when closer to the exact threshold
            closeness = 1.0 - (abs(accuracy_new - self.min_acc) / threshold_margin)
            return 500.0 * (0.5 + 0.5 * closeness)
        elif accuracy_new > (self.min_acc + threshold_margin):
            # Smaller reward for exceeding minimum by too much (we want to maximize pruning)
            return 250.0 + (accuracy_new - self.min_acc - threshold_margin) * 50
        else:
            # Severe penalty for being below minimum accuracy
            return (self.min_acc - accuracy_new) * -1000.0

    def _get_model_size_reward(self, model_size_new, param_reduction=None):
        # If param_reduction is provided, use it for a more accurate assessment
        if param_reduction is not None:
            # Scaled reward based on percentage of parameters reduced
            reduction_reward = param_reduction * 10  # Each percent reduction = 10 points
            return min(reduction_reward, 800)  # Cap at 800 points (80% reduction)

        # Fall back to the previous approach if param_reduction not provided
        delta_model_size = model_size_new - self.min_size
        if delta_model_size >= 0:
            # Penalty for exceeding target size
            return -300 * delta_model_size
        # Reward for smaller models
        reduction_factor = abs(delta_model_size) / self.min_size
        return 400 * min(reduction_factor * 2, 1.0) + 200

    def _get_comp_time_reward(self, comp_time_new, comp_time_last):
        # Calculate relative change rather than absolute
        if comp_time_last > 0:
            relative_change = (comp_time_last - comp_time_new) / comp_time_last
            return 200 * relative_change  # positive for speedups, negative for slowdowns
        return 0

    def getReward(self, accuracy, model_size, comp_time, comp_time_last, param_reduction=None):
        if not (0.99 <= self.x + self.y + self.z <= 1.01):
            raise ValueError("x, y, and z must sum to 1 or 0")
        
        accuracy_reward = self._get_accuracy_reward(accuracy)
        model_size_reward = self._get_model_size_reward(model_size, param_reduction)
        comp_time_reward = self._get_comp_time_reward(comp_time, comp_time_last)
        
        # For debugging
        print(f"Rewards - Accuracy: {accuracy_reward:.2f}, Size/Param: {model_size_reward:.2f}, Time: {comp_time_reward:.2f}")
        
        reward = (self.x*accuracy_reward) + (self.y*model_size_reward) + (self.z*comp_time_reward)
        return reward, comp_time