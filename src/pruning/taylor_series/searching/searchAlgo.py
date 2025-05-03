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
        
    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _get_model_stats(self, pruning_amount, get_accuracy=False, get_size=False, get_time=False, get_model = False, all=True):
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
        
    def meets_constraints(self, accuracy, model_size):
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
        # Using the model_size equation: -4.3237*x + 487.15 = max_size
        # Solve for x: x = (487.15 - max_size) / 4.3237
        min_pruning = (487.15 - self.max_size) / 4.3237 / 100
        print(f"Minimum pruning required for size constraint: {min_pruning:.4f}")
        return min_pruning
    
    def getReward(self, pruning_amount):
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
        meets_constraints, status = self.meets_constraints(accuracy, model_size)
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
        # Evaluate reward at current point
        reward_center = self.getReward(pruning_amount)
        
        # Evaluate reward at slightly higher pruning amount
        reward_plus = self.getReward(pruning_amount + epsilon)
        
        # Calculate gradient (forward difference)
        gradient = (reward_plus - reward_center) / epsilon
        
        print(f"Gradient at {pruning_amount:.4f}: {gradient:.6f}")
        return gradient
    
    def _run_grid_search(self, min_required_pruning, history):
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
            meets, _ = self.meets_constraints(accuracy, model_size)

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
            meets, _ = self.meets_constraints(accuracy, model_size)
            
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
            meets, _ = self.meets_constraints(accuracy, model_size)
            
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
                info = self._get_model_stats(next_pruning, all=True)
                accuracy = info['accuracy']
                model_size = info['model_size']
                meets, _ = self.meets_constraints(accuracy, model_size)
                
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
        Visualize the reward functions used in the search algorithm.
        
        Returns:
            Path to the saved reward functions plot
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
        """Run the complete search process and report results."""
        # Visualize the reward functions
        self.visualize_reward_functions()
        
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