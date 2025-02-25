from GetAccuracy import GetAccuracy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np

MIN_ACCURACY = 0.75        # Minimum acceptable accuracy
lambda_penalty = 100.0     # Penalty coefficient if accuracy < MIN_ACCURACY
lambda_model = 0.002       # Penalty coefficient for model size
lambda_compute = 0.009     # Penalty coefficient for computing time

gamma = 0.99               # Discount factor (not used much in one-step episodes)
clip_param = 0.2           # PPO clipping parameter
ppo_epochs = 4             # PPO epochs per update
batch_size = 64            # Mini-batch size for PPO update
learning_rate = 0.01      # Learning rate for policy and value networks
num_updates = 25          # Number of PPO update iterations
episodes_per_update = 8   # Episodes to collect per update

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
        s = torch.clamp(action, 0.0, 0.99)
        accuracy, model_size, computation_time = GetAccuracy(model_sel).get_accuracy(sparsity=s.item(), model_sel=model_sel, initial=False)
        penalty = max(MIN_ACCURACY - accuracy, 0.0)
        reward = s.item() - lambda_penalty * (penalty ** 2) - lambda_model * model_size - lambda_compute * computation_time
        
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
        self.act = nn.Tanh()
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            x: The input state.

        Returns:
            A tuple containing the mean and standard deviation of the action distribution.
        """
        x = self.act(self.fc1(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std)
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
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        try:
            x = self.act(self.fc1(x))
            x = self.fc2(x)
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
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_function.parameters()),
            lr = lr
        )
        
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
    
    dataset_size = states.size(0)
    
    for update in range(ppo_epochs):
        print(f"PPO Epoch: {update:03d}")
        indices = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mini_indices = indices[start:end]
            
            # Create new tensor objects for each mini-batch to avoid in-place operations
            states_mini = states[mini_indices].detach()
            actions_mini = actions[mini_indices].detach()
            log_probs_old_mini = log_probs_old[mini_indices].detach()
            returns_mini = returns[mini_indices].detach()
            
            # Move computation of values inside mini-batch loop
            log_probs, entropies, values_pred = agent.evaluate(states_mini, actions_mini)
            
            # Calculate advantages inside the loop
            advantages_mini = returns_mini - values_pred.detach()
            
            # Calculate policy loss
            ratio = torch.exp(log_probs - log_probs_old_mini)
            
            surr1 = ratio * advantages_mini
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_mini
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = nn.MSELoss()(values_pred, returns_mini)
            
            # Calculate entropy bonus
            entropy_bonus = entropies.mean()
            
            # Calculate total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            
            # Perform optimization step
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

"""Main function for training the PPO agent.

This function initializes the environment and agent, then runs the PPO training loop.
It periodically prints the training progress and final results.
"""
def main():
    print(f"Starting PPO training for {model_sel}...")

    get_acc = GetAccuracy(model_sel)

    print(f"Initial Accuracy: {get_acc.get_accuracy(sparsity=0.0, model_sel=model_sel, initial=True)[0]:.2f}")
    min_acc = input("Enter minimum acceptable accuracy: ")
    if min_acc != "":
        global MIN_ACCURACY
        MIN_ACCURACY = float(min_acc)
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

    env = PruningEnv()
    state_dim = 1
    hidden_dim = 64
    action_dim = 1
    agent = PPOAgent(state_dim, hidden_dim, action_dim, learning_rate)

    for update in range(num_updates):
        trajectories = []
        print(f"Num Update: {update:03d}")
        
        for up in range(episodes_per_update):    
            print(f"Episode: {up:03d}")
            states, actions, log_probs = [], [], []
            
            state = env.reset().float().unsqueeze(0).to(device)
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
            
        ppo_update(agent, trajectories, clip_param, ppo_epochs, batch_size)
        
        
        
        state_eval = env.reset().float().unsqueeze(0).to(device)
        mean, _ = agent.policy(state_eval)
        s_eval = torch.clamp(mean, 0.0, 0.99).item()
        accuracy, model_size, computation_time = get_acc.get_accuracy(sparsity=s_eval, model_sel=model_sel, initial=False)
        penalty = max(MIN_ACCURACY - accuracy, 0.0)
        reward = s_eval - lambda_penalty * (penalty ** 2) - lambda_model * model_size - lambda_compute * computation_time
        
        print(f"Update: {update:02d}, Sparsity: {s_eval:.4f}, Reward: {reward:.6f}, Accuracy: {accuracy:.2f}, Model Size: {model_size/(1024*1024):.2f}MB, Computation Time: {computation_time:.5f}s")
            
    print("Training completed, final sparsity: {:.2f}".format(s_eval))
    print("Final accuracy: {:.2f}".format(accuracy))
    print("Final model size: {:.2f}MB".format(model_size/(1024*1024)))
    print("Final computation time: {:.5f}s".format(computation_time))
    
if __name__ == '__main__':
    # Enable anomaly detection for debugging (uncomment if needed)
    # torch.autograd.set_detect_anomaly(True)
    main()