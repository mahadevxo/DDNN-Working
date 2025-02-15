from GetAccuracy import GetAccuracy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np

MIN_ACCURACY = 0.75        # Minimum acceptable accuracy
lambda_penalty = 100.0     # Penalty coefficient if accuracy < MIN_ACCURACY
lambda_model = 0.001       # Penalty coefficient for model size
lambda_compute = 0.001     # Penalty coefficient for computing time

gamma = 0.99               # Discount factor (not used much in one-step episodes)
clip_param = 0.2           # PPO clipping parameter
ppo_epochs = 4             # PPO epochs per update
batch_size = 32            # Mini-batch size for PPO update
learning_rate = 0.001      # Learning rate for policy and value networks
num_updates = 500          # Number of PPO update iterations
episodes_per_update = 32   # Episodes to collect per update

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

model_sel = 'vgg16'

def get_accuracy(sparsity):
    model = GetAccuracy().get_model(model_sel)
    model = GetAccuracy().prune_model(model, sparsity)
    model = GetAccuracy().fine_tuning(model)
    data_loader = GetAccuracy().get_random_images()
    accuracy, model_size, computation_time = GetAccuracy().get_accuracy(model, data_loader)
    return accuracy, model_size, computation_time

class PruningEnv:
    def __init__(self):
        self.state = torch.tensor([0.0])
        
    def reset(self):
        return self.state
    
    def step(self, action):
        s = torch.clamp(action, 0.0, 0.99)
        accuracy, model_size, computation_time = get_accuracy(s.item())
        penalty = max(MIN_ACCURACY - accuracy, 0.0)
        reward = s.item() - lambda_penalty * (penalty ** 2) - lambda_model * model_size - lambda_compute * computation_time
        
        done = True
        next_state = self.state
        info = {'accuracy': accuracy, 'model_size': model_size, 'computation_time': computation_time}
        return next_state, reward, done, info
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.tanh =(self.fc1(x))
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value_function = ValueNetwork(state_dim, hidden_dim)
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
    states = torch.cat([traj['states'] for traj in trajectories], dim=0).to(device)
    actions = torch.cat([traj['actions'] for traj in trajectories], dim=0).to(device)
    log_probs_old = torch.cat([traj['log_probs'] for traj in trajectories], dim=0).to(device)
    
    returns = torch.tensor([traj['returns'] for traj in trajectories], dtype=torch.float).to(device)
    
    values = agent.value_function(states)
    advantages = returns - values.detach()
    
    dataset_size = states.size(0)
    
    for _ in range(ppo_epochs):
        indices = np.random.permutation(dataset_size)
        