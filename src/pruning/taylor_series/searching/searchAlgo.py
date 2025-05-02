"""
Gradient-based search algorithm for finding optimal network pruning parameters.
This module implements a search strategy that combines grid search and gradient descent
to find the best pruning amount that satisfies model size and accuracy constraints.
"""

from datetime import datetime
import gc
import torch
import numpy as np
from Rewards import Reward
import matplotlib.pyplot as plt

# Device configuration
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class GradientDescentSearch:
    """
    Gradient-based search algorithm to find the optimal pruning amount
    that balances accuracy, model size, and computation time.
    """
    
    def __init__(self, min_acc=60, max_size=300):
        """
        Initialize the search algorithm.
        
        Args:
            min_acc: Minimum acceptable accuracy in percentage (default: 60)
            max_size: Maximum acceptable model size (default: 300)
        """
        # Constraint parameters
        self.min_acc = min_acc
        self.max_size = max_size
        
        # Component weights for the reward function
        self.x = 0.35  # Weight for accuracy
        self.y = 0.50  # Weight for model size
        self.z = 0.15  # Weight for computation time
        
        # Initialize reward function
        self.rewardfn = Reward(
            min_acc=self.min_acc,
            max_size=self.max_size,
            x=self.x, y=self.y, z=self.z,
        )
        
        # Tracking variables
        self.comp_time_last = None
        self.best_size = float('inf')
        
    def _clear_memory(self):
        """Release unused memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _get_accuracy(self, pruning_amount):
        """
        Calculate predicted accuracy for a given pruning amount.
        
        Args:
            pruning_amount: Amount of pruning (0.0-1.0)
            
        Returns:
            Predicted accuracy
        """
        x = pruning_amount
        y = 9.04 * x**5 - 11.10 * x**4 - 1.86 * x**3 + 4.13 * x**2 - 1.17 * x + 1.05
        return y * 75
    
    def _get_model_size(self, pruning_amount):
        """
        Calculate predicted model size for a given pruning amount.
        
        Args:
            pruning_amount: Amount of pruning (0.0-1.0)
            
        Returns:
            Predicted model size
        """
        x = pruning_amount * 100
        return -4.3237 * x + 487.15
    
    def _get_comp_time(self, pruning_amount):
        """
        Calculate predicted computation time for a given pruning amount.
        
        Args:
            pruning_amount: Amount of pruning (0.0-1.0)
            
        Returns:
            Predicted computation time
        """
        x = pruning_amount * 100
        return -0.0134 * x + 1.4681
        
    def meets_constraints(self, pruning_amount):
        """
        Check if a pruning amount meets all constraints.
        
        Args:
            pruning_amount: Amount of pruning (0.0-1.0)
            
        Returns:
            (bool, str): Tuple of (satisfies_constraints, status_message)
        """
        accuracy = self._get_accuracy(pruning_amount)
        model_size = self._get_model_size(pruning_amount)
        
        satisfies_accuracy = accuracy >= self.min_acc
        satisfies_size = model_size <= self.max_size
        
        if satisfies_accuracy and satisfies_size:
            return True, f"Valid solution: acc={accuracy:.2f}, size={model_size:.2f}"
        
        violations = []
        if not satisfies_accuracy:
            violations.append(f"Accuracy {accuracy:.2f} < {self.min_acc}")
        if not satisfies_size:
            violations.append(f"Size {model_size:.2f} > {self.max_size}")
            
        return False, "Constraints violated: " + ", ".join(violations)
    
    def calculate_required_pruning(self):
        """
        Calculate the minimum pruning required to meet the model size constraint.
        
        Returns:
            Minimum pruning amount needed
        """
        # Using the model_size equation: -4.3237*x + 487.15 = max_size
        # Solve for x: x = (487.15 - max_size) / 4.3237
        min_pruning = (487.15 - self.max_size) / 4.3237 / 100
        print(f"Minimum pruning required for size constraint: {min_pruning:.4f}")
        return min_pruning
    
    def getReward(self, pruning_amount):
        """
        Calculate the reward for a given pruning amount.
        
        Args:
            pruning_amount: Amount of pruning (0.0-1.0)
            
        Returns:
            Calculated reward value
        """
        # Check pruning amount bounds
        if pruning_amount < 0.01 or pruning_amount > 0.85:
            print(f"Pruning amount {pruning_amount:.4f} out of bounds [0.01, 0.85]")
            return -100
            
        # Calculate metrics based on pruning amount
        accuracy = self._get_accuracy(pruning_amount)
        model_size = self._get_model_size(pruning_amount)
        comp_time = self._get_comp_time(pruning_amount)
        comp_time_last = self.comp_time_last if self.comp_time_last is not None else comp_time

        # Display predicted metrics
        print(f"\nEvaluating pruning amount: {pruning_amount:.4f}")
        print(f"Predicted accuracy: {accuracy:.2f}%, model size: {model_size:.2f} "
              f"({model_size/self.max_size:.1%} of max), computation time: {comp_time:.4f}")
        
        # Check constraint satisfaction
        meets_constraints, status = self.meets_constraints(pruning_amount)
        print(status)
        
        # Track best model size for valid solutions
        if meets_constraints and model_size < self.best_size:
            self.best_size = model_size
            print(f"New smallest valid model size: {self.best_size:.2f} "
                  f"({self.best_size/self.max_size:.1%} of max)")

        # Calculate reward from the reward function
        reward = self.rewardfn.getReward(
            accuracy, model_size, comp_time, comp_time_last
        )       
        self.comp_time_last = comp_time
        
        # Add bonus for higher pruning that meets constraints
        if meets_constraints and pruning_amount > 0.45:
            pruning_bonus = (pruning_amount - 0.45) * 20
            reward += pruning_bonus
            print(f"Added high pruning bonus: +{pruning_bonus:.2f}")
        
        print(f"Final reward for pruning {pruning_amount:.4f}: {reward:.4f}")
        print('-' * 80)
        return reward
    
    def numerical_gradient(self, pruning_amount, epsilon=0.005):
        """
        Calculate numerical gradient of the reward function at a given point.
        
        Args:
            pruning_amount: Amount of pruning (0.0-1.0)
            epsilon: Small step size for finite difference (default: 0.005)
            
        Returns:
            Gradient value
        """
        # Evaluate reward at current point
        reward_center = self.getReward(pruning_amount)
        
        # Evaluate reward at slightly higher pruning amount
        reward_plus = self.getReward(pruning_amount + epsilon)
        
        # Calculate gradient (forward difference)
        gradient = (reward_plus - reward_center) / epsilon
        
        print(f"Gradient at {pruning_amount:.4f}: {gradient:.6f}")
        return gradient
    
    def _run_grid_search(self, min_required_pruning, history):
        """
        Run initial grid search to find good starting points.
        
        Args:
            min_required_pruning: Minimum pruning required for constraints
            history: Dictionary of search history to update
            
        Returns:
            Tuple of (best_reward, best_pruning_amount, best_valid_reward, 
                     best_valid_pruning, best_grid_points)
        """
        print("Phase 1: Grid Search to Find Starting Points")
        
        # Create a grid of pruning values to evaluate
        pruning_grid = np.linspace(max(0.3, min_required_pruning), 0.72, 12)
        
        # Track best solutions
        best_reward = float('-inf')
        best_pruning_amount = None
        best_valid_reward = float('-inf')
        best_valid_pruning = None
        best_grid_points = []
        
        # Evaluate each grid point
        for point in pruning_grid:
            reward = self.getReward(point)
            accuracy = self._get_accuracy(point)
            model_size = self._get_model_size(point)
            meets, _ = self.meets_constraints(point)
            
            # Update history
            history['pruning'].append(point)
            history['rewards'].append(reward)
            history['accuracy'].append(accuracy)
            history['model_size'].append(model_size)
            history['valid'].append(meets)
            history['gradients'].append(0)  # No gradient for grid search
            
            # Update best overall solution
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = point
                
            # Update best valid solution
            if meets and reward > best_valid_reward:
                best_valid_reward = reward
                best_valid_pruning = point
                best_grid_points.append((point, reward))
        
        # Sort grid points by reward
        best_grid_points.sort(key=lambda x: x[1], reverse=True)
        
        return (best_reward, best_pruning_amount, best_valid_reward, 
                best_valid_pruning, best_grid_points)
    
    def _fine_tune_search(self, best_local_point, min_required_pruning, history,
                         best_reward, best_pruning_amount, best_valid_reward, best_valid_pruning):
        """
        Perform fine-tuning around a good solution point.
        
        Args:
            best_local_point: Best point found in gradient search
            min_required_pruning: Minimum pruning required for constraints
            history: Dictionary of search history to update
            best_reward: Current best reward
            best_pruning_amount: Current best pruning amount
            best_valid_reward: Current best valid reward
            best_valid_pruning: Current best valid pruning amount
            
        Returns:
            Updated (best_reward, best_pruning_amount, best_valid_reward, best_valid_pruning)
        """
        print("\nPhase 3: Fine-tuning around best point")
        
        # Define grid of points around best local point
        fine_grid = np.linspace(
            max(min_required_pruning, best_local_point - 0.05),
            min(0.8, best_local_point + 0.05),
            7
        )
        
        for point in fine_grid:
            reward = self.getReward(point)
            accuracy = self._get_accuracy(point)
            model_size = self._get_model_size(point)
            meets, _ = self.meets_constraints(point)
            
            # Update history
            history['pruning'].append(point)
            history['rewards'].append(reward)
            history['accuracy'].append(accuracy)
            history['model_size'].append(model_size)
            history['valid'].append(meets)
            history['gradients'].append(0)
            
            # Update best overall solution
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = point
                print(f"New best reward: {best_reward:.4f} at pruning: {best_pruning_amount:.4f}")
            
            # Update best valid solution
            if meets and reward > best_valid_reward:
                best_valid_reward = reward
                best_valid_pruning = point
                print(f"New best valid solution: {best_valid_reward:.4f} at pruning: {best_valid_pruning:.4f}")
                
        return best_reward, best_pruning_amount, best_valid_reward, best_valid_pruning
    
    def search(self):
        """
        Perform the complete search for optimal pruning amount.
        
        Returns:
            Tuple of (best_pruning_amount, best_reward, history)
        """
        print("Starting Gradient Ascent Search")
        
        # Calculate minimum pruning needed for size constraint
        min_required_pruning = self.calculate_required_pruning()
        
        # Initialize search history
        history = {
            'pruning': [],
            'rewards': [],
            'accuracy': [],
            'model_size': [],
            'valid': [],
            'gradients': []
        }
        
        # Run initial grid search
        (best_reward, best_pruning_amount, best_valid_reward, 
         best_valid_pruning, best_grid_points) = self._run_grid_search(min_required_pruning, history)
        
        # Phase 2: Line search from best starting points
        print("\nPhase 2: Line Search from Best Starting Points")
        
        # Set learning parameters
        learning_rate = 0.01
        max_iterations = 50
        patience = 5
        min_gradient = 0.001
        
        # Start from best grid points
        starting_points = [point for point, _ in best_grid_points[:2]]
        if not starting_points:  # If no valid points found, use best overall point
            starting_points = [best_pruning_amount]
        
        # Try gradient descent from up to 2 starting points
        for start_point in starting_points[:2]:
            print(f"\nStarting line search from pruning amount: {start_point:.4f}")
            
            # Initialize current point
            current = start_point
            prev_reward = self.getReward(current)
            patience_counter = 0
            
            # Store initial point in history
            accuracy = self._get_accuracy(current)
            model_size = self._get_model_size(current)
            meets, _ = self.meets_constraints(current)
            
            history['pruning'].append(current)
            history['rewards'].append(prev_reward)
            history['accuracy'].append(accuracy)
            history['model_size'].append(model_size)
            history['valid'].append(meets)
            history['gradients'].append(0)
            
            # Main gradient ascent loop
            for iteration in range(max_iterations):
                print(f"\nIteration {iteration+1}/{max_iterations}")
                
                # Calculate gradient
                gradient = self.numerical_gradient(current)
                
                # Adaptive learning rate
                adaptive_lr = learning_rate / (1 + 0.1 * iteration)
                
                # Update pruning amount
                next_pruning = current + adaptive_lr * gradient
                
                # Ensure we stay within bounds
                next_pruning = np.clip(next_pruning, max(0.01, min_required_pruning), 0.8)
                
                # Calculate reward at new point
                next_reward = self.getReward(next_pruning)
                
                # Record history
                accuracy = self._get_accuracy(next_pruning)
                model_size = self._get_model_size(next_pruning)
                meets, _ = self.meets_constraints(next_pruning)
                
                history['pruning'].append(next_pruning)
                history['rewards'].append(next_reward)
                history['accuracy'].append(accuracy)
                history['model_size'].append(model_size)
                history['valid'].append(meets)
                history['gradients'].append(gradient)
                
                # Update best solutions
                if next_reward > best_reward:
                    best_reward = next_reward
                    best_pruning_amount = next_pruning
                    print(f"New best reward: {best_reward:.4f} at pruning: {best_pruning_amount:.4f}")
                
                if meets and next_reward > best_valid_reward:
                    best_valid_reward = next_reward
                    best_valid_pruning = next_pruning
                    print(f"New best valid solution: {best_valid_reward:.4f} at pruning: {best_valid_pruning:.4f}")
                
                # Check for improvement
                if next_reward <= prev_reward:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                else:
                    patience_counter = 0
                    print(f"Improved reward: {next_reward:.4f} > {prev_reward:.4f}")
                
                # Check stopping conditions
                if patience_counter >= patience:
                    print("Stopping early: No improvement for several iterations")
                    break
                
                if abs(gradient) < min_gradient:
                    print(f"Stopping early: Small gradient {gradient:.6f}")
                    break
                
                # Update current point and reward
                current = next_pruning
                prev_reward = next_reward
                
                self._clear_memory()
            
            # Perform fine search around the best point found in this run
            best_reward, best_pruning_amount, best_valid_reward, best_valid_pruning = self._fine_tune_search(
                current, min_required_pruning, history, 
                best_reward, best_pruning_amount, best_valid_reward, best_valid_pruning
            )
        
        # Return best valid solution if available, otherwise best overall solution
        if best_valid_pruning is not None:
            print("\nSearch complete - Found valid solution")
            print(f"Best valid pruning amount: {best_valid_pruning:.4f} with reward: {best_valid_reward:.4f}")
            return best_valid_pruning, best_valid_reward, history
        else:
            print("\nSearch complete - No valid solution found")
            print(f"Best pruning amount: {best_pruning_amount:.4f} with reward: {best_reward:.4f}")
            return best_pruning_amount, best_reward, history
    
    def visualize_results(self, best_pruning_amount, history):
        """
        Create visualizations of the search results.
        
        Args:
            best_pruning_amount: Best pruning amount found
            history: Dictionary containing search history
        """
        # Create a colormap based on constraint satisfaction
        colors = ['red' if not valid else 'green' for valid in history['valid']]
        
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Reward vs Pruning Amount plot
        plt.subplot(2, 2, 1)
        plt.scatter(history['pruning'], history['rewards'], c=colors, alpha=0.7)
        plt.xlabel('Pruning Amount')
        plt.ylabel('Reward')
        plt.title('Reward vs Pruning Amount')
        plt.axvline(x=best_pruning_amount, color='b', linestyle='--', 
                   label=f'Best: {best_pruning_amount:.4f}')
        plt.grid(True)
        plt.legend()
        
        # Accuracy vs Pruning Amount plot
        plt.subplot(2, 2, 2)
        plt.scatter(history['pruning'], history['accuracy'], c=colors, alpha=0.7)
        plt.axhline(y=self.min_acc, color='r', linestyle='--', label=f'Min Acc: {self.min_acc}')
        plt.xlabel('Pruning Amount')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Pruning Amount')
        plt.grid(True)
        plt.legend()
        
        # Model Size vs Pruning Amount plot
        plt.subplot(2, 2, 3)
        plt.scatter(history['pruning'], history['model_size'], c=colors, alpha=0.7)
        plt.axhline(y=self.max_size, color='r', linestyle='--', label=f'Max Size: {self.max_size}')
        plt.xlabel('Pruning Amount')
        plt.ylabel('Model Size')
        plt.title('Model Size vs Pruning Amount')
        plt.grid(True)
        plt.legend()
        
        # Gradient magnitudes or Accuracy vs Model Size
        plt.subplot(2, 2, 4)
        valid_gradients = [g for i, g in enumerate(history['gradients']) if g != 0]
        if valid_gradients:  # Only plot if we have gradients
            plt.plot(range(len(valid_gradients)), valid_gradients)
            plt.xlabel('Gradient Calculation Step')
            plt.ylabel('Gradient Magnitude')
            plt.title('Gradient Magnitudes')
            plt.grid(True)
        else:
            # Alternative plot if no gradients
            plt.scatter(history['accuracy'], history['model_size'], c=colors, alpha=0.7)
            plt.axhline(y=self.max_size, color='r', linestyle='--', label='Max Size')
            plt.axvline(x=self.min_acc, color='r', linestyle='--', label='Min Acc')
            plt.xlabel('Accuracy')
            plt.ylabel('Model Size')
            plt.title('Model Size vs Accuracy')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # Save the figure with timestamp
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig(f'res/pruning_search_results_{timenow}.png')
        plt.close()
    
    def main(self):
        """Run the complete search process and report results."""
        # Run the search
        best_pruning_amount, best_reward, history = self.search()
        
        # Print final results
        print(f"\nBest Pruning Amount: {best_pruning_amount:.4f}")
        print(f"Best Reward: {best_reward:.4f}")
        
        # Calculate final metrics
        accuracy = self._get_accuracy(best_pruning_amount)
        model_size = self._get_model_size(best_pruning_amount)
        comp_time = self._get_comp_time(best_pruning_amount)
        
        # Print final stats
        print(f"\n----------Stats at {best_pruning_amount:.4f}----------")
        print(f"Accuracy: {accuracy:.2f}, Minimum Accuracy: {self.min_acc}")
        print(f"Model Size: {model_size:.2f} ({model_size/self.max_size:.1%} of max), "
              f"Maximum Model Size: {self.max_size}")
        print(f"Computation Time: {comp_time:.4f}")
        
        # Check if constraints are satisfied
        meets, status = self.meets_constraints(best_pruning_amount)
        if meets:
            print("✅ All constraints satisfied!")
            print(f"Achieved {100 - model_size/self.max_size*100:.1f}% reduction from maximum allowed model size")
        else:
            print(f"❌ {status}")
        
        # Create visualizations
        self.visualize_results(best_pruning_amount, history)
        
        return best_pruning_amount, best_reward


if __name__ == "__main__":
    search = GradientDescentSearch()
    search.main()