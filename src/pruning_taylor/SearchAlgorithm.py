import torch
from torchvision import models
import math

class SearchAlgorithm:
    def __init__(self, original_model, min_accuracy=None,
                 init_acc = None, init_comp = None,
                 init_size = None):
        self.original_model = original_model
        self.min_accuracy = min_accuracy
        self.init_acc = init_acc
        self.init_comp = init_comp
        self.init_size = init_size
        
        #Constants for reward
        self.VERY_GOOD_REWARD_ACC = 1200   
        self.GOOD_REWARD_ACC = 101
        self.BAD_REWARD_ACC = -1024
        self.GOOD_REWARD_COMP = 13
        self.BAD_REWARD_COMP = -100
        self.GOOD_REWARD_SIZE = 7
        self.BAD_REWARD_SIZE = -10
        
    def heuristic_binary_search(self,max_iter=10):
        """
        Heuristic search to find the best pruning percentage that gets final accuracy
        as close to (but not lower than) min_accuracy, optimized for lower memory usage.
        """
        original_state = self.original_model.state_dict()
        lower, upper = 0.0, self._find_starting_point()
        best_percentage = 0.0
        best_final_acc = 0.0
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        from PruningFineTuner import PruningFineTuner
        
        for _ in range(max_iter):
            mid = (lower + upper) / 2.0
            trial_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
            trial_model.load_state_dict(original_state)
            pruner = PruningFineTuner(trial_model)
            pruner.prune(pruning_percentage=mid)
            final_acc = pruner.test(pruner.model)[0]
            if final_acc >= self.min_accuracy:
                best_percentage = mid
                best_final_acc = final_acc
                lower = mid
            else:
                upper = mid
            del pruner
            del trial_model
            if device == 'cuda':
                torch.cuda.empty_cache()
        print(f"Best pruning percentage: {best_percentage:.2f}% yields final accuracy: {best_final_acc*100:.2f}%")
        del original_state
        return best_percentage
    
    def _get_reward(self, new_acc, new_comp, new_size):
        diff_acc = new_acc - self.min_accuracy # >0 --> new is greater than min; okay ish reward
        # diff_min_acc = self.init_acc - new_acc # the more the diff, the more the reward
        diff_comp = self.init_comp - new_comp  # the more the diff, the more the reward
        diff_size = self.init_size - new_size  # the more the diff, the more the reward
        reward = 0

        if diff_acc >= 0 and diff_acc <= 0.01:
            reward += self.VERY_GOOD_REWARD_ACC * math.abs(diff_acc)
        elif diff_acc > 0.01:
            reward += self.GOOD_REWARD_ACC * math.abs(diff_acc)
        else:
            reward += self.BAD_REWARD_ACC * math.abs(diff_acc)

        if diff_comp >= 0:
            reward += self.GOOD_REWARD_COMP * math.abs(diff_comp)
        else:
            reward += self.BAD_REWARD_COMP * math.abs(diff_comp)

        if diff_size >= 0:
            reward += self.GOOD_REWARD_SIZE * math.abs(diff_size)
        else:
            reward += self.BAD_REWARD_SIZE * math.abs(diff_size)

        return reward
    
    def _find_starting_point(self):
        p = self.min_accuracy
        a = self.init_acc
        return round(a/p * 100, 3) 
    
    def heuristic_adam_search(self, max_iter=10, lr=5.0, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam-based optimization to find the best pruning percentage that gets final accuracy
        as close to (but not lower than) min_accuracy.
        """
        original_state = self.original_model.state_dict()
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

        pruning_percentage = self._find_starting_point()
        best_percentage = pruning_percentage
        best_final_acc = 0.0

        m, v = 0, 0  # Adam state variables
        for t, _ in enumerate(range(max_iter), start=1):
            trial_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
            trial_model.load_state_dict(original_state)
            from PruningFineTuner import PruningFineTuner
            pruner = PruningFineTuner(trial_model)
            pruner.prune(pruning_percentage=pruning_percentage)
            final_acc = pruner.test(pruner.model)[0]

            if final_acc >= self.min_accuracy:
                best_percentage = pruning_percentage
                best_final_acc = final_acc

            grad = -1 if final_acc > self.min_accuracy else 1

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            step = lr * m_hat / (v_hat ** 0.5 + eps)

            pruning_percentage = max(0.0, min(100.0, pruning_percentage - step))

            del pruner, trial_model
            if device == 'cuda':
                torch.cuda.empty_cache()

        return self._print_accuracy(best_percentage, best_final_acc, original_state)

    def new_algo(self):
        '''
        loss function would
        where
        accuracy with max importance
        next computation time
        then model size
        
        also if org acc is 50% and we need to bring it down to 20%
        then we compute the ratio between then 50/20 * 100 = 25
        we start with a pruning ratio of 25% rather than start at the mid
        25% may not be the best option but it converges faster
        
        reward function for accuracy:
        if acc > threshold: reward -0.2 * (acc - threshold)
        if acc ≈ threshold: reward 10
        if acc < threshold: reward -1000
        
        reward function fot computation time:
        if comp_time < threshold: reward 10
        else reward: -100
        
        reward function for model size:
        if model_size < org model size: reward 10
        else reward: -10
        
        but then we have lets say a set of best_percentages
        lets say [20.0, 20.3, 20.5, 21.04]
        we can then take the one with the lowest loss function
        we can store the last 20 accuracies, comp times and model size in a list
        [pruning amount, acc, computation time, model size] or we can have the reward function instead
        
        + 0.25 * comp_time + 0.05 * model_size
        '''
        
        best_percentage = 0.0
        
        return