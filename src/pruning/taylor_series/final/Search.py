from operator import itemgetter
import gc
from Rewards import Reward, GetResults
from PruningFineTuner import PruningFineTuner
from Pruning import Pruning
from copy import deepcopy
import torch
import numpy as np
from heapq import nsmallest
from MVCNN_Trainer import MVCNN_Trainer
import random
import math

class Search:
    def __init__(self, model, min_acc, min_size, acc_imp=0.33, comp_time_imp=0.33, size_imp=0.33):
        self.model = model
        self.min_acc = min_acc
        self.min_size = min_size
        self.reward = Reward(min_acc, min_size)
        self.comp_time_mean = 0
        self.comp_time_min = 0
        self.model_size_min = 0
        self.rewards = -1*(np.inf)
        self.getResults = GetResults()
        print("Getting filter ranks")
        self.ranks = PruningFineTuner(model=self.model, 
                                      train_amt=0.1, test_amt=0.1).prune(
                                          rank_filters=True)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.rewardfn = Reward(min_acc, min_size)
        self.mvcnntrainer = MVCNN_Trainer(optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001), train_amt=0.1, test_amt=0.1)
        self.initial_accuracy = self.mvcnntrainer.get_val_accuracy(self.model)[1]
        print(f"Initial Accuracy: {self.initial_accuracy}")
        self.x = acc_imp
        self.y = comp_time_imp
        self.z = size_imp
        print(f"Acc: {self.x}, Comp Time: {self.y}, Size: {self.z}")
        self.old_num_filters = self.mvcnntrainer.get_num_filters(self.model)
    
    def _init_csv(self):
        with open('results.csv', 'w') as f:
            f.write('pruning amount, pre accuracy, actual fine tune, fine tuned, compute time, model size\n')
        
    def _write_to_csv(self, pruning_amount, pre_accuracy, actual_fine_tune, fine_tuned, comp_time, model_size):
        with open('results.csv', 'a') as f:
            f.write(f"{pruning_amount}, {pre_accuracy}, {actual_fine_tune}, {fine_tuned}, {comp_time}, {model_size}\n")
    
    def _reset(self):
        # Clear previous data structures to prevent memory leaks
        if hasattr(self, 'filter_ranks'):
            for key in list(self.filter_ranks.keys()):
                del self.filter_ranks[key]
        if hasattr(self, 'activations'):
            for act in self.activations:
                del act
            
        self.filter_ranks = {}
        self.activations = []
        self.gradients = []
        self.activation_to_layer = {}
        self.grad_index = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def get_pruning_plan(self, n):
        n = int(n)
        filters_to_prune =  nsmallest(n, self.ranks, key=itemgetter(2))
        print(f"Pruning {len(filters_to_prune)} for {n} filters")
        
        filters_to_pruner_per_layer = {}
        for (layer_n, f, _) in filters_to_prune:
            if layer_n not in filters_to_pruner_per_layer:
                filters_to_pruner_per_layer[layer_n] = []
            filters_to_pruner_per_layer[layer_n].append(f)
        
        
        for layer_n in filters_to_pruner_per_layer:
            filters_to_pruner_per_layer[layer_n] = sorted(filters_to_pruner_per_layer[layer_n])
            for i in range(len(filters_to_pruner_per_layer[layer_n])):
                filters_to_pruner_per_layer[layer_n][i] = filters_to_pruner_per_layer[layer_n][i] - i
        
        
        filters_to_prune = []
        for layer_n in filters_to_pruner_per_layer:
            for i in filters_to_pruner_per_layer[layer_n]:
                filters_to_prune.append((layer_n, i))
        
        self._reset()
        return filters_to_prune
    
    def prune_and_get_rewards(self, pruning_amount, model, actual_fine_tune):
        prune_targets = self.get_pruning_plan(pruning_amount)
        pruner = Pruning(model=model)
        actual_fine_tune = True
        for idx, (layer_index, filter_index) in enumerate(prune_targets):
            model = pruner.prune_conv_layers(model, layer_index=layer_index, filter_index=filter_index)
            
            if idx % 3 == 0:
                self._clear_memory()
        
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                if layer.weight is not None:
                    layer.weight.data = layer.weight.data.float()
                if layer.bias is not None:
                    layer.bias.data = layer.bias.data.float()
        
        model = model.to(self.device)
        self._clear_memory()
        
        print(f"Number of Filters {self.old_num_filters} --> {self.mvcnntrainer.get_num_filters(model)} for Pruning Amount {pruning_amount}")
        
        accuracy_pre_fine_tuning, comp_time, model_size = self.getResults.get_results(model)
        if actual_fine_tune:
            print("Not Approximating Accuracy")
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            mvcnntrainer = MVCNN_Trainer(optimizer)
            _, fine_tuned = mvcnntrainer.fine_tune(model)
        
        else:
            print("Approximating Accuracy")
            fine_tuned = self.getResults.get_approx_acc(pruning_amount, accuracy_pre_fine_tuning, self.initial_accuracy).item()
        
        reward = self.rewardfn.getReward(
            accuracy=fine_tuned,
            model_size=model_size,
            comp_time=comp_time,
            comp_time_last=self.comp_time_mean,
            x = self.x, 
            y = self.y, 
            z = self.z
        )
        
        print(f"Pruning Amount: {pruning_amount}, Reward: {reward}, Accuracy: {fine_tuned}, Comp time: {comp_time}, Model size: {model_size}")
        self._write_to_csv(pruning_amount, accuracy_pre_fine_tuning, actual_fine_tune, fine_tuned, comp_time, model_size)
        self._reset()
        return reward
            
    def adam_gradient(
    self, 
    initial_pruning=30, 
    learning_rate=0.1, 
    beta1=0.9, 
    beta2=0.999, 
    epsilon=1e-8, 
    max_iter=30, 
    delta=1e-1, 
    clip_range=(1.0, 99.0), 
    x=0.33, 
    y=0.33, 
    z=0.33,
    patience=5,
    decay=0.95):
        pruning_amount = initial_pruning
        m, v = 0.0, 0.0
        best_pruning_amount = pruning_amount
        best_reward = -np.inf
        no_improve_steps = 0

        actual_fine_tune = False
        self._init_csv()

        for t in range(1, max_iter + 1):
            print(f"\n=== Iteration {t} ===")
            if t > int(max_iter * 0.5):
                actual_fine_tune = True
                print("Enabling fine-tuning")

            # Slightly reduce LR to stabilize convergence
            lr = learning_rate * (decay ** t)

            # Gradient estimation using central difference
            prune_plus = min(pruning_amount + delta, clip_range[1])
            prune_minus = max(pruning_amount - delta, clip_range[0])

            model_copy_1 = deepcopy(self.model).to(self.device)
            r_plus = self.prune_and_get_rewards(prune_plus, model_copy_1, actual_fine_tune=False)

            model_copy_2 = deepcopy(self.model).to(self.device)
            r_minus = self.prune_and_get_rewards(prune_minus, model_copy_2, actual_fine_tune=False)

            grad = (r_plus - r_minus) / (2 * delta)

            # Adam moment updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Update pruning amount with Adam
            pruning_amount += lr * m_hat / (np.sqrt(v_hat) + epsilon)
            pruning_amount = float(torch.clamp(torch.tensor(pruning_amount), *clip_range))

            # Evaluate reward with updated pruning amount
            model_copy_final = deepcopy(self.model).to(self.device)
            reward_now = self.prune_and_get_rewards(pruning_amount, model_copy_final, actual_fine_tune=actual_fine_tune)

            # Track best reward
            if reward_now > best_reward + 1e-4:
                best_reward = reward_now
                best_pruning_amount = pruning_amount
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            print(f"Pruning: {pruning_amount:.4f}, Reward: {reward_now:.4f}, Best: {best_reward:.4f}")

            # Early stopping if not improving
            if no_improve_steps >= patience:
                print(f"Early stopping: no improvement in {patience} steps")
                break

        return self.reset_show_output(
            best_pruning_amount, best_reward
        )
    
    def simulated_annealing(self, initial_pruning=5, max_iter=30, delta=1e-3, clip_range=(1.00, 99.00), temperature=1.0, cooling=0.95, patience=5):
        from copy import deepcopy  # if not already imported
        current_pruning = initial_pruning
        current_reward = self.prune_and_get_rewards(current_pruning, deepcopy(self.model).to(self.device), actual_fine_tune=False)
        best_pruning = current_pruning
        best_reward = current_reward
        no_improve_steps = 0

        self._init_csv()

        for t in range(1, max_iter + 1):
            print(f"\n=== SA Iteration {t}, Temperature: {temperature:.4f} ===")
            actual_fine_tune = t > int(max_iter * 0.5)
            if actual_fine_tune:
                print("Enabling fine-tuning")

            perturbation = random.uniform(-delta, delta)
            candidate_pruning = current_pruning + perturbation
            candidate_pruning = float(torch.clamp(torch.tensor(candidate_pruning), *clip_range).item())

            candidate_reward = self.prune_and_get_rewards(candidate_pruning, deepcopy(self.model).to(self.device), actual_fine_tune)

            delta_reward = candidate_reward - current_reward
            accept_prob = 1.0 if delta_reward > 0 else math.exp(delta_reward / temperature)
            rand_val = random.random()

            if rand_val < accept_prob:
                print(f"Accepted candidate: pruning {candidate_pruning:.4f}, reward {candidate_reward:.4f} (delta: {delta_reward:.4f}, prob: {accept_prob:.4f})")
                current_pruning = candidate_pruning
                current_reward = candidate_reward
                no_improve_steps = 0
                if candidate_reward > best_reward:
                    best_reward = candidate_reward
                    best_pruning = candidate_pruning
            else:
                print(f"Rejected candidate: pruning {candidate_pruning:.4f}, reward {candidate_reward:.4f} (delta: {delta_reward:.4f}, prob: {accept_prob:.4f})")
                no_improve_steps += 1

            temperature *= cooling
            print(f"Current: pruning {current_pruning:.4f}, reward {current_reward:.4f}, Best: {best_pruning:.4f}, reward {best_reward:.4f}")

            if no_improve_steps >= patience:
                print(f"Early stopping: no improvement in {patience} iterations")
                break

        return self.reset_show_output(best_pruning, best_reward)
    
    def hill_climbing(self, initial_pruning=5, step_size=0.5, max_iter=50, clip_range=(1.0, 99.0), tolerance=1e-4, fine_tune=True):
        current_pruning = initial_pruning
        best_pruning = current_pruning
        model_copy = deepcopy(self.model).to(self.device)
        current_reward = self.prune_and_get_rewards(current_pruning, model_copy, actual_fine_tune=fine_tune)
        iteration = 0
        improved = True
        while improved and iteration < max_iter:
            improved = False
            candidates = [current_pruning + step_size, current_pruning - step_size]
            for candidate in candidates:
                candidate = float(torch.clamp(torch.tensor(candidate), *clip_range).item())
                model_copy = deepcopy(self.model).to(self.device)
                candidate_reward = self.prune_and_get_rewards(candidate, model_copy, actual_fine_tune=fine_tune)
                if candidate_reward > current_reward + tolerance:
                    current_pruning = candidate
                    current_reward = candidate_reward
                    best_pruning = candidate
                    improved = True
            iteration += 1
            print(f"Hill Climbing Iteration {iteration}: current pruning {current_pruning:.4f}, reward {current_reward:.4f}")
        return self.reset_show_output(best_pruning, current_reward)
    
    def reset_show_output(self, arg0, best_reward):
        self._reset()
        print(f"\n==> Best Pruning Amount: {arg0:.4f}, Best Reward: {best_reward:.4f}")
        return arg0, best_reward

    def __del__(self):
        self._reset()
        print("Reset Search object deleted and memory cleared.")