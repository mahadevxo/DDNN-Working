from datetime import datetime
import gc
import torch
import numpy as np
from Rewards import Reward
import matplotlib.pyplot as plt
from infogetter import InfoGetter

# Device configuration
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

class GradientDescentSearch:
    """
    Implements a gradient-based search algorithm to optimize model pruning under accuracy and size constraints.

    This class manages the search process, including grid search, gradient ascent, and fine-tuning, 
    to find the optimal pruning amount that maximizes a custom reward function while satisfying user-defined constraints.
    """
    def __init__(self):
        # Constraint parameters
        self.min_acc = float(input("Enter minimum accuracy (default 60): ") or 60)
        self.max_size = float(input("Enter maximum model size (default 300): ") or 300)
        
        # Component weights for the reward function
        self.x = float(input("Enter weight for accuracy (default 0.35): ") or 0.35)        # Weight for accuracy
        self.y = float(input("Enter weight for model size (default 0.50): ") or 0.50)      # Weight for model size
        self.z = float(input("Enter weight for computation time (default 0.15): ") or 0.15)# Weight for computation time 
        
        # Initialize reward function
        self.rewardfn = Reward(
            min_acc=self.min_acc,
            max_size=self.max_size,
            x=self.x, y=self.y, z=self.z,
        )
        
        # Tracking variables
        self.comp_time_last = None
        self.best_size = float('inf')
        self.use_eq = True
        
    def _clear_memory(self):
        """
        Clears GPU/MPS memory by forcing garbage collection and emptying caches.
        
        This function helps prevent memory leaks during the search process by
        explicitly freeing unused memory after operations.
        
        Args:
            None
            
        Returns:
            None
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _get_model_stats(self, pruning_amount, get_accuracy=False, get_size=False, get_time=False, get_model=False, all=True):
        """
        Retrieves statistics about the pruned model with the given pruning amount.
        
        This function calls the InfoGetter to prune the model at the specified amount
        and return various metrics about the resulting model.
        
        Args:
            pruning_amount: Float between 0-1 specifying the percentage of filters to prune
            get_accuracy: If True, only returns the model accuracy
            get_size: If True, only returns the model size
            get_time: If True, only returns the computation time
            get_model: If True, only returns the pruned model
            all: If True, returns all metrics (model, accuracy, size, and computation time)
            
        Returns:
            Dictionary with model metrics or specific metric based on the flags
        """
                
        if self.use_eq:
            x = pruning_amount
            acc = (9.04 * x**5 - 11.10 * x**4 - 1.86 * x**3 + 4.13 * x**2 - 1.17 * x + 1.05)*75
            size = -4.3237 * x *100 + 487.15
            time = -0.0134 * x * 100 + 1.4681
            return {
                'model': None,
                'accuracy': acc,
                'model_size': size,
                'computation_time': time
            }
        infogettr = InfoGetter()
        info = infogettr.getInfo(pruning_amount)
        self._clear_memory()
        
        if all:
            model, acc, size, time = info[0], info[1], info[3], info[2]
            return {
                'model': model,
                'accuracy': acc,
                'model_size': size,
                'computation_time': time
            }
        if get_model:
            return info[0]
        if get_accuracy:
            return info[1]
        if get_time:
            return info[2]
        if get_size:
            return info[3]
        
    def _meets_constraints(self, accuracy, model_size):
        """
        Checks if the pruned model satisfies the user-defined constraints.
        
        Evaluates whether the pruned model's accuracy and size meet the 
        minimum accuracy and maximum size requirements.
        
        Args:
            accuracy: The accuracy of the pruned model (percentage)
            model_size: The size of the pruned model (MB)
            
        Returns:
            Tuple of:
              - Boolean indicating if all constraints are met
              - String message describing the result or violations
        """
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
    
    def getReward(self, pruning_amount):
        """
        Calculates the reward for a given pruning amount.
        
        Prunes the model at the specified amount, evaluates its metrics, and calculates
        a reward value using the reward function, considering accuracy, model size, and
        computation time. Also applies bonuses for solutions that effectively satisfy 
        constraints with higher pruning amounts.
        
        Args:
            pruning_amount: Float between 0-1 specifying the percentage of filters to prune
            
        Returns:
            Float representing the calculated reward value
        """
        # Check pruning amount bounds
        if pruning_amount < 0.01 or pruning_amount > 0.85:
            print(f"Pruning amount {pruning_amount:.4f} out of bounds [0.01, 0.85]")
            return -100
        
        # Calculate metrics based on pruning amount
        info = self._get_model_stats(pruning_amount, all=True)
        accuracy = info['accuracy']
        model_size = info['model_size']
        comp_time = info['computation_time']
        comp_time_last = self.comp_time_last if self.comp_time_last is not None else comp_time

        # Display predicted metrics
        print(f"\nEvaluating pruning amount: {pruning_amount:.4f}")
        print(f"Accuracy: {accuracy:.2f}%, model size: {model_size:.2f} "
              f"({model_size/self.max_size:.1%} of max), computation time: {comp_time:.4f}")
        
        # Check constraint satisfaction
        meets_constraints, status = self._meets_constraints(accuracy, model_size)
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
    
    def _numerical_gradient(self, pruning_amount, epsilon=0.005):
        """
        Calculates the numerical gradient of the reward function at the given pruning amount.
        
        Uses a forward difference approximation to compute the gradient, which helps
        determine the direction to adjust the pruning amount to increase the reward.
        
        Args:
            pruning_amount: Current pruning percentage to evaluate
            epsilon: Small step size for numerical differentiation
            
        Returns:
            Float representing the gradient of the reward function at the given point
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
        Performs grid search to find good starting points for gradient-based optimization.
        
        Evaluates the reward function at evenly spaced pruning amounts to identify
        promising regions in the search space. This provides starting points for the
        more refined gradient-based search.
        
        Args:
            min_required_pruning: Minimum pruning percentage needed to meet size constraint
            history: Dictionary tracking the search history (pruning amounts, rewards, etc.)
            
        Returns:
            Tuple containing:
              - Best overall reward
              - Best overall pruning amount
              - Best reward among valid solutions
              - Best pruning amount among valid solutions
              - List of best grid points sorted by reward
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
            
            info = self._get_model_stats(point, all=True)
            accuracy = info['accuracy']
            model_size = info['model_size']
            meets, _ = self._meets_constraints(accuracy, model_size)

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
                best_grid_points.append((best_valid_pruning, best_valid_reward))
        # Sort grid points by reward
        best_grid_points.sort(key=lambda x: x[1], reverse=True)

        return (best_reward, best_pruning_amount, best_valid_reward, 
                best_valid_pruning, best_grid_points)
    
    def _fine_tune_search(self, best_local_point, min_required_pruning, history,
                         best_reward, best_pruning_amount, best_valid_reward, best_valid_pruning):
        """
        Performs fine-tuning around the best point found during gradient search.
        
        Creates a finer grid of pruning amounts around the best point identified
        so far and evaluates each point to refine the optimal solution.
        
        Args:
            best_local_point: The pruning amount that performed best in gradient search
            min_required_pruning: Minimum pruning required to satisfy size constraint
            history: Dictionary tracking the search history
            best_reward: Best reward found so far
            best_pruning_amount: Best pruning amount found so far
            best_valid_reward: Best reward among valid solutions found so far
            best_valid_pruning: Best pruning amount among valid solutions found so far
            
        Returns:
            Updated values for best_reward, best_pruning_amount, best_valid_reward, and best_valid_pruning
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
            info = self._get_model_stats(point, all=True)
            accuracy = info['accuracy']
            model_size = info['model_size']
            meets, _ = self._meets_constraints(accuracy, model_size)
            
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
    
    def calculate_required_pruning(self):
        """
        Estimates the minimum pruning amount needed to satisfy the size constraint.
        
        Since the linear equation relating pruning amount to model size is not available,
        this function uses a conservative estimate based on the maximum size constraint.
        It starts with a modest pruning percentage that can be adjusted during the search.
        
        Args:
            None
            
        Returns:
            Float representing the estimated minimum required pruning (between 0-1)
        """
        # Start with a conservative minimum pruning of 30%
        # The actual required pruning will be found during search
        min_pruning = 0.05
        print(f"Using estimated minimum pruning of {min_pruning:.4f} (30%)")
        return min_pruning

    def search(self):  # sourcery skip: low-code-quality
        """
        Executes the complete search algorithm to find optimal pruning amount.
        
        Combines grid search, gradient ascent, and fine-tuning to efficiently
        explore the search space and find the pruning amount that maximizes
        the reward function while satisfying constraints.
        
        Args:
            None
            
        Returns:
            Tuple containing:
              - The optimal pruning amount
              - The reward value for the optimal amount
              - Dictionary containing the search history
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
        starting_points = [point for point, _ in best_grid_points[:2]] or [best_pruning_amount]
        
        # Try gradient descent from up to 2 starting points
        for start_point in starting_points[:2]:
            print(f"\nStarting line search from pruning amount: {start_point:.4f}")
            
            # Initialize current point
            current = start_point
            prev_reward = self.getReward(current)
            patience_counter = 0
            
            # Store initial point in history
            info = self._get_model_stats(current, all=True)
            accuracy = info['accuracy']
            model_size = info['model_size']
            meets, _ = self._meets_constraints(accuracy, model_size)
            
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
                gradient = self._numerical_gradient(current)
                
                # Adaptive learning rate
                adaptive_lr = learning_rate / (1 + 0.1 * iteration)
                
                # Update pruning amount
                next_pruning = current + adaptive_lr * gradient
                
                # Ensure we stay within bounds
                next_pruning = np.clip(next_pruning, max(0.01, min_required_pruning), 0.8)
                
                # Calculate reward at new point
                next_reward = self.getReward(next_pruning)
                
                # Record history
                info = self._get_model_stats(next_pruning, all=True)
                accuracy = info['accuracy']
                model_size = info['model_size']
                meets, _ = self._meets_constraints(accuracy, model_size)
                
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
        Visualizes the search process and results as plots.
        
        Creates multiple plots showing the relationship between pruning amount
        and various metrics (reward, accuracy, model size, and gradients), 
        highlighting valid and invalid solutions and the best found solution.
        
        Args:
            best_pruning_amount: The optimal pruning amount found
            history: Dictionary containing the search history
            
        Returns:
            None, but saves the visualization to a file
        """
        # Create a colormap based on constraint satisfaction
        colors = ['green' if valid else 'red' for valid in history['valid']]

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
        if valid_gradients := [g for g in history['gradients'] if g != 0]:
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
    
    def visualize_reward_functions(self):
        """
        Visualizes the reward functions used in the search algorithm.
        
        Creates plots showing how the reward components (accuracy, model size,
        computation time) behave across their respective input ranges.
        
        Args:
            None
            
        Returns:
            String path to the saved visualization file
        """
        print("Plotting reward functions...")
        
        # Create save path
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        save_path = f'res/reward_functions_{timenow}.png'
        
        # Plot and save the reward functions
        self.rewardfn.plot_reward_functions(save_path)
        
        print(f"Reward functions plot saved to {save_path}")
        return save_path
    
    def main(self):
        """
        Runs the complete search process and reports the final results.
        
        Executes the search algorithm, displays the results, and creates
        visualizations to help understand the search process and outcome.
        
        Args:
            None
            
        Returns:
            Tuple containing:
              - The optimal pruning amount
              - The reward value for the optimal amount
        """
        # Visualize the reward functions
        self.visualize_reward_functions()
        
        # Run the search
        best_pruning_amount, best_reward, history = self.search()
        
        # Print final results
        print(f"\nBest Pruning Amount: {best_pruning_amount:.4f}")
        print(f"Best Reward: {best_reward:.4f}")
        
        # Calculate final metrics
        info = self._get_model_stats(best_pruning_amount, all=True)
        accuracy = info['accuracy']
        model_size = info['model_size']
        comp_time = info['computation_time']
        
        # Print final stats
        print(f"\n----------Stats at {best_pruning_amount:.4f}----------")
        print(f"Accuracy: {accuracy:.2f}, Minimum Accuracy: {self.min_acc}")
        print(f"Model Size: {model_size:.2f} ({model_size/self.max_size:.1%} of max), "
              f"Maximum Model Size: {self.max_size}")
        print(f"Computation Time: {comp_time:.4f}")
        
        # Check if constraints are satisfied
        meets, status = self._meets_constraints(accuracy, model_size)
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