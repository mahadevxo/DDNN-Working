import numpy as np
import matplotlib.pyplot as plt

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

    def _get_accuracy_reward(self, acc, peak_reward=5.0):
        """
        Calculate accuracy reward using a Gaussian curve.
        Reward peaks at minimum accuracy and gradually decreases on either side.
        
        Args:
            acc: Accuracy value (0.0-1.0)
            peak_reward: Maximum reward value at the peak
            
        Returns:
            Gaussian-based reward value for accuracy
        """
        # Parameters for the Gaussian curve
        mu = self.min_acc  # Center at minimum acceptable accuracy
        sigma_below = 0.05  # Standard deviation for below min_acc
        sigma_above = 0.10  # Standard deviation for above min_acc
        
        # Use different sigmas depending on whether accuracy is below or above the minimum
        sigma = sigma_below if acc < self.min_acc else sigma_above
        
        # Gaussian function: f(x) = a * exp(-(x - mu)^2 / (2 * sigma^2))
        gaussian_reward = peak_reward * np.exp(-((acc - mu) ** 2) / (2 * sigma ** 2))
        
        # Apply penalty for accuracies below minimum
        if acc < self.min_acc:
            penalty_factor = 1 - 5 * (self.min_acc - acc)  # Linear penalty increases as acc decreases
            return gaussian_reward * penalty_factor
        
        return gaussian_reward
    
    def _get_model_size_reward(self, model_size_new, max_reward=5.0):
        """
        Calculate model size reward using a Gaussian curve.
        Reward peaks at the smallest model size and decreases as size increases.
        
        Args:
            model_size_new: Size of the model
            max_reward: Maximum possible reward for smallest models
            
        Returns:
            Gaussian-based reward value for model size
        """
        # Parameters for the Gaussian curve
        mu = 0  # Center at size = 0 (ideal smallest model)
        sigma = self.max_size * 0.5  # Standard deviation, allowing good rewards for models well below max_size
        
        # Gaussian function: f(x) = a * exp(-(x - mu)^2 / (2 * sigma^2))
        gaussian_reward = max_reward * np.exp(-((model_size_new - mu) ** 2) / (2 * sigma ** 2))
        
        # Apply penalty for exceeding maximum size
        if model_size_new > self.max_size:
            excess_ratio = model_size_new / self.max_size - 1.0
            penalty = 4.0 * np.tanh(3.0 * excess_ratio)
            return gaussian_reward - penalty
        
        return gaussian_reward

    def _get_comp_time_reward(self, comp_time_new, comp_time_last, max_reward=3.0):
        """
        Calculate computation time reward using a Gaussian function.
        Reward peaks at zero computation time and decreases as time increases.
        
        Args:
            comp_time_new: New computation time
            comp_time_last: Previous computation time
            max_reward: Maximum possible reward
            
        Returns:
            Gaussian-based reward value for computation time
        """
        # Parameters for the Gaussian curve
        mu = 0  # Center at time = 0 (ideal fastest model)
        sigma = comp_time_last * 0.7  # Scale sigma relative to previous computation time
        
        # Gaussian function: f(x) = a * exp(-(x - mu)^2 / (2 * sigma^2))
        gaussian_reward = max_reward * np.exp(-((comp_time_new - mu) ** 2) / (2 * sigma ** 2))
        
        # Add bonus for improvement over last time
        if comp_time_new < comp_time_last:
            improvement_ratio = (comp_time_last - comp_time_new) / comp_time_last
            bonus = 2.0 * np.tanh(3.0 * improvement_ratio)
            return gaussian_reward + bonus
        
        return gaussian_reward

    def plot_reward_functions(self, save_path=None):
        """
        Plot the reward functions to visualize how they behave across different input values.
        
        Args:
            save_path: Path to save the figure (if None, the plot is displayed)
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot accuracy reward function
        acc_values = np.linspace(0.5, 1.0, 100)  # Accuracy from 50% to 100%
        acc_rewards = [self._get_accuracy_reward(acc) for acc in acc_values]
        
        axs[0].plot(acc_values * 100, acc_rewards)  # Convert to percentage for readability
        axs[0].axvline(x=self.min_acc * 100, color='r', linestyle='--', 
                      label=f'Min Acc: {self.min_acc*100:.1f}%')
        axs[0].set_title('Accuracy Reward Function')
        axs[0].set_xlabel('Accuracy (%)')
        axs[0].set_ylabel('Reward')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot model size reward function
        size_values = np.linspace(0, self.max_size * 1.5, 100)  # Include values above max_size
        size_rewards = [self._get_model_size_reward(size) for size in size_values]
        
        axs[1].plot(size_values, size_rewards)
        axs[1].axvline(x=self.max_size, color='r', linestyle='--', 
                      label=f'Max Size: {self.max_size}')
        axs[1].set_title('Model Size Reward Function')
        axs[1].set_xlabel('Model Size')
        axs[1].set_ylabel('Reward')
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot computation time reward function
        # Assume a reference comp_time_last of 1.0
        reference_time = 1.0
        time_values = np.linspace(0, 2.0, 100)  # From 0 to 2x the reference time
        time_rewards = [self._get_comp_time_reward(t, reference_time) for t in time_values]
        
        axs[2].plot(time_values, time_rewards)
        axs[2].axvline(x=reference_time, color='r', linestyle='--', 
                      label=f'Reference Time: {reference_time}')
        axs[2].set_title('Computation Time Reward Function')
        axs[2].set_xlabel('Computation Time')
        axs[2].set_ylabel('Reward')
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return fig

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
        size_r = self._get_model_size_reward(model_size_new=model_size)
        time_r = self._get_comp_time_reward(comp_time_new=comp_time, comp_time_last=self.comp_time_last)
        
        # Print component rewards for debugging
        print(f"Component Rewards - Accuracy: {acc_r:.2f}, Size: {size_r:.2f}, Time: {time_r:.2f}")

        # Store current computation time for next iteration
        self.comp_time_last = comp_time

        # Signal constraint violations, but avoid harsh penalties that disrupt convergence
        if accuracy < self.min_acc:
            print(f"CONSTRAINT VIOLATION: Accuracy {accuracy:.2f} below minimum {self.min_acc*100:.2f}")
            
        if model_size > self.max_size:
            print(f"CONSTRAINT VIOLATION: Model size {model_size:.2f} exceeds maximum {self.max_size:.2f}")
        
        # Add smooth synergy bonus for solutions that are both accurate and small
        # Use continuous functions rather than hard thresholds
        if accuracy >= self.min_acc:
            # Smooth bonus that increases as model gets smaller (relative to max size)
            size_ratio = model_size / self.max_size
            if size_ratio < 1.0:
                # Bonus proportional to how much smaller than max_size
                synergy_factor = 1.0 - size_ratio
                synergy_bonus = 2.0 * synergy_factor * (1.0 - np.exp(-(accuracy - self.min_acc) * 50))
                print(f"Adding synergy bonus: +{synergy_bonus:.2f}")
                
                # Apply bonus smoothly to both components
                acc_r += synergy_bonus * 0.4
                size_r += synergy_bonus * 0.6

        # Calculate weighted sum of rewards
        raw_reward = (self.x * acc_r) + (self.y * size_r) + (self.z * time_r)
        
        # Apply smoother clipping using tanh instead of hard boundaries
        # This prevents abrupt gradient changes at the boundaries
        clipped_reward = 8.0 * np.tanh(raw_reward / 8.0)
        
        # Apply smoothing for stability across iterations
        self.reward = (1 - self.smooth_factor) * clipped_reward + self.smooth_factor * self.prev_reward
        self.prev_reward = self.reward
        
        print(f"Final Reward: {self.reward:.4f}")
        return self.reward