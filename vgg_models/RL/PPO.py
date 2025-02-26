from GetAccuracy import GetAccuracy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import torch.nn.functional as F

MIN_ACCURACY = 0.75        # Minimum acceptable accuracy
lambda_penalty = 500.0     # Penalty coefficient if accuracy < MIN_ACCURACY
lambda_model = 0.001       # Penalty coefficient for model size
lambda_compute = 0.005     # Penalty coefficient for computing time

gamma = 0.99               # Discount factor (not used much in one-step episodes)
clip_param = 0.2           # PPO clipping parameter
ppo_epochs = 5             # PPO epochs per update
batch_size = 32            # Mini-batch size for PPO update
learning_rate = 0.001      # Reduced learning rate for stability
num_updates = 30           # Number of PPO update iterations
episodes_per_update = 12   # Episodes to collect per update

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

model_sel = input("Enter model selection (vgg16, vgg11, vgg19, alexnet): ")

print(f"Selected model: {model_sel}")


class PruningEnv:
    def __init__(self):
        self.state = torch.tensor([0.0])
        
    def reset(self):
        return self.state
    
    def step(self, action):
        """Takes a step in the environment.

        Calculates the reward based on the action (sparsity), accuracy, model size, and computation time.

        Args:
            action: The sparsity value (action).

        Returns:
            A tuple containing the next state, reward, done flag, and info dictionary.
        """
        s = torch.clamp(action, 0.01, 0.90)  # Avoid extreme values close to 0
        accuracy, model_size, computation_time = GetAccuracy(model_sel).get_accuracy(sparsity=s.item(), model_sel=model_sel, initial=False)
        penalty = max(MIN_ACCURACY - accuracy, 0.0)
        accuracy_reward = 0 if penalty == 0 else -lambda_penalty * np.exp(penalty * 2)
        sparsity_reward = s.item() * 2.0
        size_penalty = lambda_model * model_size
        compute_penalty = lambda_compute * computation_time
        
        reward = accuracy_reward + sparsity_reward - size_penalty - compute_penalty
        
        done = True
        next_state = self.state
        info = {'accuracy': accuracy, 'model_size': model_size, 'computation_time': computation_time}
        return next_state, reward, done, info
    
class PolicyNetwork(nn.Module):
    """Policy network for PPO agent.

    This network takes the state as input and outputs the mean and standard deviation of a normal
    distribution from which the action is sampled.
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        """Initializes the PolicyNetwork.

        Args:
            state_dim: Dimension of the state space.
            hidden_dim: Dimension of the hidden layer.
            action_dim: Dimension of the action space.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Use ReLU instead of Tanh for better gradient flow
        self.act = nn.LeakyReLU()
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # Initialize log_std with a safer value
        self.log_std = nn.Parameter(torch.ones(action_dim)*-2.0)
        
        # Initialize weights with a better method
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization for better convergence."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc_mean.bias)

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            x: The input state.

        Returns:
            A tuple containing the mean and standard deviation of the action distribution.
        """
        # Add gradient checking
        if torch.isnan(x).any():
            print("NaN detected in input to policy network")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            
        x = self.act(self.fc1(x))
        mean = torch.sigmoid(self.fc_mean(x)) * 0.9  # Use sigmoid to constrain to (0,0.9)
        std = torch.exp(torch.clamp(self.log_std, -20, 0))
        
        # Check for NaNs
        if torch.isnan(mean).any():
            print("NaN detected in policy network output mean")
            mean = torch.where(torch.isnan(mean), torch.ones_like(mean) * 0.5, mean)
        
        return mean, std
    
class ValueNetwork(nn.Module):
    """Value network for PPO agent.

    This network takes the state as input and outputs the estimated value of that state.
    """
    def __init__(self, state_dim, hidden_dim):
        """Initializes the ValueNetwork.

        Args:
            state_dim: Dimension of the state space.
            hidden_dim: Dimension of the hidden layer.
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Use ReLU for better gradient propagation
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization for better convergence."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        try:
            # Check for NaNs
            if torch.isnan(x).any():
                print("NaN detected in input to value network")
                x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                
            x = self.act1(self.fc1(x))
            x = self.act2(self.fc2(x))
            x = self.fc3(x)
            
            # Check for NaNs in output
            if torch.isnan(x).any():
                print("NaN detected in value network output")
                x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
                
            return x
        except Exception as exp:
            print(x, "Exception: ", exp)
            exit()

class PPOAgent:
    """PPO agent for pruning a neural network.

    This agent uses PPO to learn a policy that selects the optimal sparsity for pruning a neural network.
    """
    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        """Initializes the PPOAgent.

        Args:
            state_dim: Dimension of the state space.
            hidden_dim: Dimension of the hidden layer.
            action_dim: Dimension of the action space.
            lr: Learning rate for the optimizer.
        """
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.value_function = ValueNetwork(state_dim, hidden_dim).to(device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=lr*2, eps=1e-5)
        
    def select_action(self, state):
        mean, std = self.policy(state)
        dist = distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate(self, state, action):
        mean, std = self.policy(state)
        dist = distributions.Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_function(state)
        return log_prob, entropy, value
    
def ppo_update(agent, trajectories, clip_param, ppo_epochs, batch_size):
    """Updates the policy and value networks using PPO.

    This function implements the PPO algorithm to update the agent's policy and value networks
    based on collected trajectories.

    Args:
        agent: The PPO agent.
        trajectories: A list of trajectories, where each trajectory is a dictionary containing
            'states', 'actions', 'log_probs', 'returns', and 'info'.
        clip_param: The PPO clipping parameter.
        ppo_epochs: The number of epochs to train the PPO for.
        batch_size: The mini-batch size for training.
    """
    states = torch.cat([traj['states'] for traj in trajectories], dim=0).to(device)
    actions = torch.cat([traj['actions'] for traj in trajectories], dim=0).to(device)
    log_probs_old = torch.cat([traj['log_probs'] for traj in trajectories], dim=0).to(device)
    
    returns = torch.tensor([traj['returns'] for traj in trajectories], dtype=torch.float).unsqueeze(1).to(device)
    
    # Check for NaNs in input data
    if torch.isnan(states).any() or torch.isnan(actions).any() or torch.isnan(log_probs_old).any() or torch.isnan(returns).any():
        print("NaN detected in input data to PPO update")
        # Replace NaNs with zeros
        states = torch.where(torch.isnan(states), torch.zeros_like(states), states)
        actions = torch.where(torch.isnan(actions), torch.zeros_like(actions), actions)
        log_probs_old = torch.where(torch.isnan(log_probs_old), torch.zeros_like(log_probs_old), log_probs_old)
        returns = torch.where(torch.isnan(returns), torch.zeros_like(returns), returns)
    
    dataset_size = states.size(0)
    
    for update in range(ppo_epochs):
        print(f"PPO Epoch: {update:03d}")
        indices = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)  # Prevent out of bounds
            if start >= end:  # Skip if we've exhausted the dataset
                continue
                
            mini_indices = indices[start:end]
            
            # Create new tensor objects for each mini-batch to avoid in-place operations
            states_mini = states[mini_indices].detach()
            actions_mini = actions[mini_indices].detach()
            log_probs_old_mini = log_probs_old[mini_indices].detach()
            returns_mini = returns[mini_indices].detach()
            
            # Move computation of values inside mini-batch loop
            log_probs, entropies, values_pred = agent.evaluate(states_mini, actions_mini)
            
            # Calculate advantages inside the loop with better numerical stability
            advantages_mini = returns_mini - values_pred.detach()
            if advantages_mini.std() > 1e-8:
                advantages_mini = (advantages_mini - advantages_mini.mean()) / (advantages_mini.std() + 1e-8)
            
            # Calculate policy loss with numerical safety checks
            ratio = torch.exp(log_probs - log_probs_old_mini)
            ratio = torch.clamp(ratio, 0.01, 100.0)  # Prevent extreme ratio values
            
            surr1 = ratio * advantages_mini
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_mini
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Check for NaN in policy loss
            if torch.isnan(policy_loss).any():
                print("NaN detected in policy loss")
                continue  # Skip this batch
            
            # Calculate value loss
            value_loss = F.mse_loss(values_pred, returns_mini)
            
            # Check for NaN in value loss
            if torch.isnan(value_loss).any():
                print("NaN detected in value loss")
                continue  # Skip this batch
            
            # Calculate entropy bonus
            entropy_bonus = entropies.mean()        
            
            # Calculate total loss
            agent.policy_optimizer.zero_grad()
            policy_total_loss = policy_loss - 0.01 * entropy_bonus  # Reduced entropy coefficient
            policy_total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)
            agent.policy_optimizer.step()
            
            # Value update
            agent.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.value_function.parameters(), 0.5)
            agent.value_optimizer.step()


"""Main function for training the PPO agent.

This function initializes the environment and agent, then runs the PPO training loop.
It periodically prints the training progress and final results.
"""

def get_min_acc():
    print(f"Starting PPO training for {model_sel}...")

    get_acc = GetAccuracy(model_sel)

    print(f"Initial Accuracy: {get_acc.get_accuracy(sparsity=0.0, model_sel=model_sel, initial=True)[0]:.2f}")
    min_acc = input("Enter minimum acceptable accuracy (Default: 0.75): ")
    if min_acc != "":
        global MIN_ACCURACY
        MIN_ACCURACY = float(min_acc)
        

def training():
    env = PruningEnv()
    state_dim = 1
    hidden_dim = 64
    action_dim = 1
    agent = PPOAgent(state_dim, hidden_dim, action_dim, learning_rate)
    get_acc = GetAccuracy(model_sel)
    
    best_sparsity = 0.0
    best_accuracy = 0.0
    best_reward = float('-inf')
    best_model_size = 0.0
    best_comp_time = 0.0
    
    file_path = f"{model_sel}_results.csv"
    with open(file_path, 'w') as f:
        f.write("Sparsity,Accuracy,Model Size,Computation Time,Reward\n")

    for update in range(num_updates):
        trajectories = []
        print(f"Num Update: {update:03d}")
        
        for up in range(episodes_per_update):    
            print(f"Episode: {up:03d}")
            states, actions, log_probs = [], [], []
            
            state = env.reset().float().unsqueeze(0).to(device)
            # Add try-except to catch potential distribution errors
            try:
                action, log_prob = agent.select_action(state)
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                
                next_state, reward, done, info = env.step(action)
                trajectory = {
                    'states': torch.cat(states, dim=0),
                    'actions': torch.cat(actions, dim=0),
                    'log_probs': torch.cat(log_probs, dim=0),
                    'returns': reward,
                    'info': info
                }
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error in episode {up}: {e}")
                continue
            
        if len(trajectories) > 0:  # Only update if we have valid trajectories
            try:
                ppo_update(agent, trajectories, clip_param, ppo_epochs, batch_size)
            except Exception as e:
                print(f"Error in PPO update: {e}")
        else:
            print("No valid trajectories for update")
            continue
        
        # Evaluation with error handling
        try:
            state_eval = env.reset().float().unsqueeze(0).to(device)
            mean, _ = agent.policy(state_eval)
            s_eval = torch.clamp(mean, 0.01, 0.90).item()
            accuracy, model_size, computation_time = get_acc.get_accuracy(sparsity=s_eval, model_sel=model_sel, initial=False)
            penalty = max(MIN_ACCURACY - accuracy, 0.0)
            accuracy_reward = 0 if penalty == 0 else -lambda_penalty * np.exp(penalty * 2)
            sparsity_reward = s_eval * 2.0
            size_penalty = lambda_model * model_size
            compute_penalty = lambda_compute * computation_time
            reward = sparsity_reward + accuracy_reward - size_penalty - compute_penalty
            
            with open(file_path, 'a') as f:
                f.write(f"{s_eval:.2f},{accuracy:.2f},{model_size/(1024*1024):.2f},{computation_time:.5f},{reward:.2f}\n")
            
            if accuracy >= MIN_ACCURACY and (
                (s_eval > best_sparsity and abs(accuracy - best_accuracy) < 1.0) or  # Better sparsity with similar accuracy
                reward > best_reward  # Better overall reward
            ):
                best_sparsity = s_eval
                best_accuracy = accuracy
                best_reward = reward
                best_model_size = model_size
                best_comp_time = computation_time
                print(f"New best result found! Sparsity: {best_sparsity:.2f}, Accuracy: {best_accuracy:.2f}, Model Size: {best_model_size/(1024*1024):.2f}MB, Comp Time: {best_comp_time:.5f}s")
        except Exception as e:
            print(f"Error in evaluation: {e}")
            
    return s_eval, accuracy, model_size, computation_time, best_sparsity, best_accuracy, best_model_size, best_comp_time, best_reward
        
def main():
    print(f"Minimum acceptable accuracy: {MIN_ACCURACY:.2f}")

    print(f"Settings:\n \
    Lambda Penalty: {lambda_penalty},\n \
    Lambda Model: {lambda_model},\n \
    Lambda Compute: {lambda_compute},\n \
    Gamma: {gamma},\n \
    Clip Param: {clip_param},\n \
    PPO Epochs: {ppo_epochs},\n \
    Batch Size: {batch_size},\n \
    Learning Rate: {learning_rate},\n \
    Num Updates: {num_updates},\n \
    Episodes per Update: {episodes_per_update},\n \
    Device: {device}\n")

    try:
        training_results = training()
        s_eval, accuracy, model_size, computation_time = training_results[:4]
        best_sparsity, best_accuracy, best_model_size, best_comp_time, best_reward = training_results[4:]
                
        print("\n--- Best Results ---")
        print("Best sparsity: {:.2f}".format(best_sparsity))
        print("Best accuracy: {:.2f}".format(best_accuracy))
        print("Best model size: {:.2f}MB".format(best_model_size/(1024*1024)))
        print("Best computation time: {:.5f}s".format(best_comp_time))
        print("Best reward: {:.6f}".format(best_reward))
        
        # Also show final results
        print("\n--- Final Results ---")
        print("Final sparsity: {:.2f}".format(s_eval))
        print("Final accuracy: {:.2f}".format(accuracy))
        print("Final model size: {:.2f}MB".format(model_size/(1024*1024)))
        print("Final computation time: {:.5f}s".format(computation_time))
    except Exception as e:
        print(f"Error in main training loop: {e}")
    
if __name__ == '__main__':
    get_min_acc()
    main()