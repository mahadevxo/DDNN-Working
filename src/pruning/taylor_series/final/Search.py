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
        
        accuracy_pre_fine_tuning, comp_time, model_size = self.getResults.get_results(model)
        if actual_fine_tune:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            mvcnntrainer = MVCNN_Trainer(optimizer)
            _, fine_tuned = mvcnntrainer.fine_tine(model)
        
        else:
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
            
    def adam_gradient(self, initial_pruning=0.01, learning_rate=0.05, 
                      beta1=0.9, beta2=0.999, epsilon=1e-8, 
                      max_iter=30, delta=1e-3, 
                      clip_range=(0.01, 0.99), x=0.33, y=0.33, z=0.33):
        
        pruning_amount = initial_pruning
        m, v = 0.0, 0.0
        best_pruning_amount = pruning_amount
        best_reward = -(np.inf)
        
        
        actual_fine_tune=False
        self._init_csv()
        for t in range(1, max_iter+1):
            print(f"Starting Iteration {t}")
            if t > int(max_iter/1.5):
                actual_fine_tune=True
                print("Fine tuning model")
            
            model_new = deepcopy(self.model)
            model_new.to(self.device)
            r_plus = self.prune_and_get_rewards(pruning_amount+delta, model_new, actual_fine_tune=False)
            r_minus = self.prune_and_get_rewards(pruning_amount-delta, model_new, actual_fine_tune=False)
            
            grad = (r_plus - r_minus) / (2 * delta)
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            pruning_amount = pruning_amount.numpy()
            pruning_amount += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            pruning_amount = np.clip(pruning_amount, clip_range[0], clip_range[1])
            pruning_amount = torch.tensor(pruning_amount, device=self.device)
            
            reward_now = self.prune_and_get_rewards(pruning_amount.item(), model_new, actual_fine_tune=actual_fine_tune)
            if reward_now > best_reward:
                best_reward = reward_now
                best_pruning_amount = pruning_amount.item()
            
            print(f"Iteration {t}, Pruning Amount: {pruning_amount}, Reward: {reward_now}")
        print(f"Best Pruning Amount: {best_pruning_amount}, Best Reward: {best_reward}")
        self._reset()
        return best_pruning_amount, best_reward
    
    def __del__(self):
        self._reset()
        print("Reset Search object deleted and memory cleared.")