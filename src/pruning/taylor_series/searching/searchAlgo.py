from datetime import datetime
import gc
import torch
import cma
import numpy as np
from Rewards import Reward
import matplotlib.pyplot as plt

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
class cmaesSearch:
    def __init__(self):
        self.min_acc = 70
        self.max_size = 300
        # Update weights to give more importance to accuracy
        self.x = 0.33  # Increased weight for accuracy (was 0.8)
        self.y = 0.33  # Increased weight for model size (was 0.1)
        self.z = 0.33  # Same weight for computation time
        
        self.rewardfn = Reward(
            min_acc = self.min_acc,
            max_size = self.max_size,
            x = self.x, y = self.y, z = self.z,
        )
        self.comp_time_last = None
        
    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def _get_accuracy(self, x):
        y = 9.04 * x**5 - 11.10 * x**4 - 1.86 * x**3 + 4.13 * x**2 - 1.17 * x + 1.05
        y*=75
        return y
    
    def _get_model_size(self, x):
        x*=100
        return -4.3237*x + 487.15
    
    def _get_comp_time(self, x):
        x*=100
        return -0.0134*x + 1.4681
    
    def getReward(self, pruning_amount):
        # Set stricter upper boundary for pruning
        if pruning_amount > 0.8:  # Changed from 0.85 to 0.8
            print("Pruning amount too high")
            return -100000

        if not (0.99 <= self.x + self.y + self.z <= 1.01):
            raise ValueError("x, y, and z must sum to 1 or 0")
        
        accuracy = self._get_accuracy(pruning_amount)
        model_size = self._get_model_size(pruning_amount)
        comp_time = self._get_comp_time(pruning_amount)
        comp_time_last = self.comp_time_last if self.comp_time_last is not None else 0

        # Calculate base reward from the reward function
        reward = self.rewardfn.getReward(
            accuracy, model_size, comp_time, comp_time_last
        )       
        self.comp_time_last = comp_time
        
        # Add strongly negative reward for accuracy below minimum
        if accuracy < self.min_acc:
            penalty = -10000 * (self.min_acc - accuracy)**2
            reward += penalty
            print(f"Applied accuracy below minimum penalty: {penalty}")
            return reward  # Exit early with severe penalty
        
        # Add more balanced sweet spot bonus
        accuracy_margin = 2.0  # Wider margin (was 1.0)
        
        if self.min_acc <= accuracy < self.min_acc+accuracy_margin:
            closeness_factor = 1.0 - (accuracy-self.min_acc) / accuracy_margin
            # Smaller bonus for high pruning
            sweet_spot_bonus = pruning_amount * 200 * (0.5 + 0.5 * closeness_factor)  # Reduced from 350
            reward += sweet_spot_bonus
            print(f"Added accuracy sweet spot bonus: +{sweet_spot_bonus}")
        
        elif accuracy >= self.min_acc and pruning_amount >= 0.5:
            # Reduced bonus to prevent over-pruning
            reward += pruning_amount * 150  # Reduced from 350
            print(f"Added accuracy bonus: +{pruning_amount * 150}")
        
        # Much stronger penalty for excessive pruning when accuracy drops
        if accuracy < self.min_acc + 1.5 and pruning_amount > 0.7:
            penalty = (pruning_amount - 0.7) * 1000
            reward -= penalty
            print(f"Applied excessive pruning penalty: -{penalty}")
        
        # Increase penalty for wasted capacity 
        elif accuracy > self.min_acc + 3.0:
            excess = accuracy - (self.min_acc + 3.0)
            wasted_capacity = min(400, excess * 500)  # Increased from 200/400
            reward -= wasted_capacity
            print(f"Applied wasted capacity penalty: -{wasted_capacity}")
        
        print(f"Accuracy: {accuracy}, Model Size: {model_size}, Comp Time: {comp_time}, Pruning Amount: {pruning_amount}, Reward: {reward}")
        print('-'*100)
        return reward
    
    def search(self):
        print("Starting Search Algorithm")
        
        try:
            # Lower initial guess and sigma to avoid always hitting upper bound
            es: cma.EvolutionStrategy = cma.CMAEvolutionStrategy(
                [0.3],  # Start from 0.3 instead of 0.05
                0.2,    # Slightly reduced sigma
                {
                    'bounds': [0.000, 0.8],  # Lower upper bound from 0.95 to 0.8
                    'maxiter': 25,
                    'tolx': 1e-2,
                    'popsize': 6,
                    'verbose': 1
                }
            )
        except Exception as e:
            print(f"Error while initializing CMA-ES: {e}")
            print("Falling back to simpler initialization")
            
            es = cma.CMAEvolutionStrategy(
                [0.3],  # Start from 0.3 instead of 0.05
                0.2,
                {
                    'bounds': [0.05, 0.8],  # Lower upper bound from 0.95 to 0.8
                    'verbose': 1
                }
            )
            
        best_reward: float = float(-np.inf)
        best_pruning_amount: float = None
        
        history: dict = {
            'pruning': [],
            'rewards': [],
            'accuracy': []
        }
        
        promising_regions: list = []
        stagnation_counter: int = 0
        prev_best_reward: float = float(-np.inf)
        
        print("-------------------------Starting with Binary Search-------------------------")
        # Update binary search points to include more points in the middle range
        binary_search_points = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75  # Added more points in middle, max at 0.75
        ]
        binary_rewards = []
        
        for point in binary_search_points:
            print(f"Binary Search Point: {point}")
            reward = self.getReward(point)
            binary_rewards.append((point, reward))
            
            history['pruning'].append(point)
            history['rewards'].append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = point
                print(f"New best reward: {best_reward} at pruning amount: {best_pruning_amount}")
            
        binary_rewards.sort(key=lambda x: x[1], reverse=True)
        
        for i, (point, reward) in enumerate(binary_rewards[:2]):
            width = 0.15
            min_bound = max(0.05, point - width/2)
            max_bound = min(0.95, point + width/2)
            
            for offset in [-0.07, -0.03, 0.03, 0.07]:
                new_point = point + offset
                if min_bound <= new_point <= max_bound:
                    promising_regions.append((max(0.05, new_point - 0.05), 
                                              min(0.85, new_point + 0.05)))
        
        if best_pruning_amount is not None:
            best_region_min = max(0.05, best_pruning_amount - 0.12)
            best_region_max = min(0.95, best_pruning_amount + 0.12)
            initial_points = np.linspace(
                best_region_min, best_region_max, num=5
            )
        else:
            initial_points = np.linspace(0.05, 0.95, num=5)
        print(f"Initial Points: {initial_points}")
        
        print("-------------------------Initial Target Exploration-------------------------")
        for point in initial_points:
            print(f"Evaluating point: {point}")
            reward = self.getReward(point)
            
            history['pruning'].append(point)
            history['rewards'].append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_pruning_amount = point
                print(f"New best reward: {best_reward} at pruning amount: {best_pruning_amount}")
                
                promising_regions.append((max(0.05, point - 0.05), min(0.85, point + 0.05)))
        
        self._clear_memory()
        
        iteration = 1
        max_iterations = 100
        improvement_threshold = 5.0
        
        def explore_region(center, width=0.1, steps=7):
            nonlocal best_reward, best_pruning_amount
            
            print(f"Expoloring region: {center} with width: {width}")
            min_bound = max(0.05, center - width/2)
            max_bound = min(0.95, center + width/2)
            
            points = []
            points.append(center)
            
            for step in range(1, (steps//2)+1):
                offset = width * step/steps-1*0.8
                points.append(center - offset)
                points.append(center + offset)
                
            for point in sorted(points):
                if min_bound <= point <= max_bound:
                    print(f"Trying Fine-grained point: {point}")
                    reward = self.getReward(point)
                    
                    history['pruning'].append(point)
                    history['rewards'].append(reward)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_pruning_amount = point
                        print(f"New best reward: {best_reward} at pruning amount: {best_pruning_amount}")
                        
                        if reward > prev_best_reward + improvement_threshold:
                            return True
            self._clear_memory()
            return False
        
        while not es.stop() and iteration <= max_iterations:
            print(f"--------------------------Iteration {iteration}-------------------------")
            try:
                solutions = es.ask()
                rewards = []
                
                print(f"Evaluating {len(solutions)} solutions")
                for x in solutions:
                    pruning_amount = x[0]
                    print(f"Evaluating solution: {x}")
                    
                    reward = self.getReward(pruning_amount)
                    rewards.append(reward)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_pruning_amount = pruning_amount
                        print(f"New best reward: {best_reward} at pruning amount: {best_pruning_amount}")
                        
                        if reward > prev_best_reward + improvement_threshold:
                            promising_regions.append((max(0.05, pruning_amount - 0.05), min(0.85, pruning_amount + 0.05)))
                            
                    history['pruning'].append(pruning_amount)
                    history['rewards'].append(reward)
                    
                    self._clear_memory()
                    
                if not solutions or rewards:
                    print("No solutions or rewards found. Skipping this iteration.")
                    break
                
                try:
                    es.tell(solutions, rewards)
                    if abs(prev_best_reward - best_reward) < 1e-3:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                    
                    prev_best_reward = best_reward
                    
                    if stagnation_counter >= 3:
                        print("Stagnation detected. Trying Alternative Exploration")
                        
                        if promising_regions:
                            region = promising_regions.pop(0)
                            center = (region[0] + region[1]) / 2
                            width =  region[1] - region[0]
                            explore_region(center, width=width, steps=5)
                            stagnation_counter = 0
                        else:
                            print("Restarting search")
                            if best_pruning_amount is not None:
                                
                                es = cma.CMA.CMAEvolutionStrategy(
                                    [best_pruning_amount],
                                    0.15,
                                    {
                                        'bounds': [0.00, 0.9],
                                        'verbose': 1
                                    }
                                )
                                stagnation_counter = 0
                except Exception as e:
                    print(f"Error during CMA-ES iteration: {e}")
                    print("Falling back to simpler initialization")
                    
                    if iteration == 1:
                        print("Falling back to grid search")
                        grid_pruning_amounts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        grid_rewards = []
                        
                        for amount in grid_pruning_amounts:
                            print(f"Evaluating grid point: {amount}")
                            reward = self.getReward(amount)
                            grid_rewards.append((amount, reward))
                            
                            if reward > best_reward:
                                best_reward = reward
                                best_pruning_amount = amount
                                print(f"New best reward: {best_reward} at pruning amount: {best_pruning_amount}")
                        print(f"Best grid point: {best_pruning_amount} with reward: {best_reward}")
                    else:
                        print("Continuing with results so far")
                    break
            except Exception as e:
                print(f"Error during CMA-ES iteration: {e}")
                print("Continuing with results so far")
                break
            
            print(f"Iteration {iteration} completed")
            print(f"Mean Pruning Amount: {np.mean([x[0] for x in solutions])}")
            print(f"Best Pruning Amount: {best_pruning_amount}")
            print(f"Best Reward: {best_reward}")
            print(f"CMA-ES sigma: {es.sigma}")
            
            iteration += 1
            self._clear_memory()
            
        print("Search Algorithm Completed")
        print(f"Best Pruning Amount: {best_pruning_amount}")
        print(f"Best Reward: {best_reward}")
        return best_pruning_amount, best_reward, history
    
    def main(self):
        best_pruning_amount, best_reward, history = self.search()
        print(f"Best Pruning Amount: {best_pruning_amount}")
        print(f"Best Reward: {best_reward}")
        
        print(f"----------Stats at {best_pruning_amount}----------")
        print(f"Accuracy: {self._get_accuracy(best_pruning_amount)}, Minimum Accuracy: {self.min_acc}")
        print(f"Model Size: {self._get_model_size(best_pruning_amount)}, Maximum Model Size: {self.max_size}")
        print(f"Computation Time: {self._get_comp_time(best_pruning_amount)}")
        
        plt.figure(figsize=(12, 8))
        plt.scatter(history['pruning'], history['rewards'], c='blue', label='Rewards')
        plt.xlabel('Pruning Amount')
        plt.ylabel('Reward')
        plt.title('Reward vs Pruning Amount')
        plt.tight_layout()
        
        plt.axvline(x=best_pruning_amount, color='r', linestyle='--', label=f'Best Pruning Amount: {best_pruning_amount:.4f}')
        plt.axhline(y=best_reward, color='g', linestyle='--', label=f'Best Reward: {best_reward:.2f}')
        plt.legend()
        plt.grid()
        #time in DD-MM-YYYY-HH-MM-SS
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        plt.savefig(f'res/pruning_search_results_{timenow}.png')
        plt.close()

if __name__ == "__main__":
    search = cmaesSearch()
    search.main()