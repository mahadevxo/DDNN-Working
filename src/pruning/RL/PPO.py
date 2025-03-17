from GetAccuracy import GetAccuracy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np

MIN_ACCURACY = 0.75        # Minimum acceptable accuracy
lambda_penalty = 1000.0    # Penalty coefficient if accuracy < MIN_ACCURACY
lambda_model = 0.0005      # Penalty coefficient for model size
lambda_compute = 0.05      # Penalty coefficient for computing time

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
        self.min_action = MIN_ACCURACY
        
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
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Add an extra layer
        # Use LeakyReLU instead of ReLU to prevent dying neurons
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # Start with a smaller log_std for more deterministic behavior initially
        self.log_std = nn.Parameter(torch.ones(action_dim)*-3.0)
        
        # Initialize weights with a better method
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for LeakyReLU."""
        nn.init.kaiming_normal_(self.fc1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_mean.bias)

    def forward(self, x):
        # Add gradient checking
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))  # Pass through the additional layer
        
        # Use different activation for mean to ensure better action bounds
        # Sigmoid * 0.89 + 0.01 ensures range of (0.01, 0.9)
        mean = torch.sigmoid(self.fc_mean(x)) * 0.89 + 0.01
        
        # Use adaptive std that decreases over time for better exploration->exploitation
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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization for better convergence."""
        nn.init.kaiming_normal_(self.fc1.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
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
            raise

class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.value_function = ValueNetwork(state_dim, hidden_dim).to(device)
        
        # Use a more conservative learning rate with Adam
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5, weight_decay=1e-5)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=lr*2, eps=1e-5, weight_decay=1e-5)
        
        # Add learning rate schedulers for better convergence
        self.policy_scheduler = optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=10, gamma=0.9)
        self.value_scheduler = optim.lr_scheduler.StepLR(self.value_optimizer, step_size=10, gamma=0.9)
        
    def select_action(self, state):
        # More robust action selection
        with torch.no_grad():  # No need for gradients during action selection
            mean, std = self.policy(state)
            dist = distributions.Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0.01, 0.9)  # Enforce bounds here too
            log_prob = dist.log_prob(action)
        return action, log_prob
    
    def evaluate(self, state, action):
        mean, std = self.policy(state)
        dist = distributions.Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_function(state)
        return log_prob, entropy, value
    
    def step_schedulers(self):
        """Step the learning rate schedulers"""
        self.policy_scheduler.step()
        self.value_scheduler.step()


    
def compute_advantages(returns, initial_values):
    advantages = returns - initial_values
    if advantages.std() > 1e-8:  # Check for zero variance
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def compute_policy_loss(log_probs, log_probs_old, advantages, clip_param):
    ratio = torch.exp(log_probs - log_probs_old)
    ratio = torch.clamp(ratio, 0.01, 10.0)  # Prevent extreme values
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    return -torch.min(surr1, surr2).mean()

def update_policy(agent, policy_loss, entropy_bonus, entropy_coeff):
    agent.policy_optimizer.zero_grad()
    policy_total_loss = policy_loss - (entropy_coeff * entropy_bonus)
    policy_total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 0.5)  # Gradient clipping
    agent.policy_optimizer.step()

def update_value(agent, values_pred, returns):
    value_crit = nn.SmoothL1Loss()  # Huber loss for value function
    value_loss = value_crit(values_pred, returns)
    agent.value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.value_function.parameters(), 0.5)  # Gradient clipping
    agent.value_optimizer.step()
    
def prepare_data(trajectories):
    states = torch.cat([traj['states'] for traj in trajectories], dim=0).to(device)
    actions = torch.cat([traj['actions'] for traj in trajectories], dim=0).to(device)
    log_probs_old = torch.cat([traj['log_probs'] for traj in trajectories], dim=0).to(device)
    returns = torch.tensor([traj['returns'] for traj in trajectories], dtype=torch.float).unsqueeze(1).to(device)

    states = torch.nan_to_num(states)
    actions = torch.nan_to_num(actions)
    log_probs_old = torch.nan_to_num(log_probs_old)
    returns = torch.nan_to_num(returns)
    return states, actions, log_probs_old, returns

def update_mini_batch(agent, states_mini, actions_mini, log_probs_old_mini, returns_mini, initial_values_mini, clip_param, epoch, ppo_epochs):
    log_probs, entropies, values_pred = agent.evaluate(states_mini, actions_mini)

    advantages_mini = compute_advantages(returns_mini, initial_values_mini)
    policy_loss = compute_policy_loss(log_probs, log_probs_old_mini, advantages_mini, clip_param)

    if torch.isnan(policy_loss).any():  # Check for NaNs
        print("NaN detected in policy loss")
        return

    entropy_coeff = max(0.005, 0.01 * (1 - epoch / ppo_epochs))  # Adaptive entropy coefficient
    entropy_bonus = entropies.mean()

    update_policy(agent, policy_loss, entropy_bonus, entropy_coeff)
    update_value(agent, values_pred, returns_mini)


def ppo_update(agent, trajectories, clip_param, ppo_epochs, batch_size):
    states, actions, log_probs_old, returns = prepare_data(trajectories)
    dataset_size = states.size(0)

    with torch.no_grad():
        initial_values = agent.value_function(states)

    for epoch in range(ppo_epochs):
        print(f"PPO Epoch: {epoch:03d}")
        indices = np.random.permutation(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            if start >= end:
                continue

            mini_indices = indices[start:end]

            states_mini = states[mini_indices].detach()
            actions_mini = actions[mini_indices].detach()
            log_probs_old_mini = log_probs_old[mini_indices].detach()
            returns_mini = returns[mini_indices].detach()
            initial_values_mini = initial_values[mini_indices].detach()

            update_mini_batch(agent, states_mini, actions_mini, log_probs_old_mini, returns_mini, initial_values_mini, clip_param, epoch, ppo_epochs)


def get_min_acc():
    print(f"Starting PPO training for {model_sel}...")

    get_acc = GetAccuracy(model_sel)

    print(f"Initial Accuracy: {get_acc.get_accuracy(sparsity=0.0, model_sel=model_sel, initial=True)[0]:.5f}")
    min_acc = int(input("Enter minimum acceptable accuracy (Default: 75): "))


    if min_acc != "":
        if min_acc > 1:
            min_acc /= 100

        global MIN_ACCURACY
        MIN_ACCURACY = float(min_acc)
    print(f"Minimum acceptable accuracy: {MIN_ACCURACY:.5f}")


def collect_trajectories(env, agent, episodes_per_update):
    trajectories = []
    for episode in range(episodes_per_update):
        print(f"Episode: {episode:03d}")
        try:
            trajectory = run_episode(env, agent)
            trajectories.append(trajectory)
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            continue  # Skip to the next episode if there's an error
    return trajectories

def run_episode(env, agent):
    states, actions, log_probs = [], [], []
    state = env.reset().float().unsqueeze(0).to(device)
    done = False
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        state = next_state.float().unsqueeze(0).to(device)

    return {
        'states': torch.cat(states, dim=0),
        'actions': torch.cat(actions, dim=0),
        'log_probs': torch.cat(log_probs, dim=0),
        'returns': reward,  # Use the final reward as the return for the episode
        'info': info
    }

def evaluate_agent(env, agent, get_acc):
    num_evals = 3
    eval_results = []
    for _ in range(num_evals):
        state_eval = env.reset().float().unsqueeze(0).to(device)
        with torch.no_grad():
            mean, _ = agent.policy(state_eval)
        s_eval = torch.clamp(mean, 0.01, 0.90).item()
        accuracy, model_size, computation_time = get_acc.get_accuracy(sparsity=s_eval, model_sel=model_sel, initial=False)
        reward = calculate_reward(accuracy, s_eval, model_size, computation_time)
        eval_results.append((s_eval, accuracy, model_size, computation_time, reward))

    eval_results.sort(key=lambda x: x[4])
    return eval_results[num_evals // 2]  # Return median result

def calculate_reward(accuracy, sparsity, model_size, computation_time):
    accuracy_factor = min(1.0, accuracy / MIN_ACCURACY)
    accuracy_reward = 2.0 * accuracy_factor - 1.0

    if accuracy < MIN_ACCURACY:
        accuracy_penalty = lambda_penalty * (1.0 - accuracy_factor) ** 2
        accuracy_reward -= accuracy_penalty

    sparsity_reward = sparsity * 2.0
    size_penalty = lambda_model * model_size / (1024 * 1024)
    compute_penalty = lambda_compute * computation_time * 100

    return accuracy_reward + sparsity_reward - size_penalty - compute_penalty

def save_best_model(agent, best_sparsity, best_accuracy, best_reward, best_model_size, best_comp_time):
    checkpoint = {
        'policy_state_dict': agent.policy.state_dict(),
        'value_state_dict': agent.value_function.state_dict(),
        'sparsity': best_sparsity,
        'accuracy': best_accuracy,
        'reward': best_reward,
        'model_size': best_model_size,
        'comp_time': best_comp_time
    }
    torch.save(checkpoint, f"{model_sel}_best_model.pt")
    print("Best model saved!")

def training():
    env = PruningEnv()
    state_dim = 1
    hidden_dim = 128  # Increased hidden dimension for more capacity
    action_dim = 1
    learning_rate = 0.0005  # Reduced learning rate for better stability
    agent = PPOAgent(state_dim, hidden_dim, action_dim, learning_rate)
    get_acc = GetAccuracy(model_sel)

    best_sparsity = 0.0
    best_accuracy = 0.0
    best_reward = float('-inf')
    best_model_size = 0.0
    best_comp_time = 0.0

    # Early stopping parameters
    patience = 5
    no_improvement_count = 0

    file_path = f"{model_sel}_results.csv"
    with open(file_path, 'w') as f:
        f.write("Update,Sparsity,Accuracy,Model Size,Computation Time,Reward\n")

    for update in range(num_updates):
        
        for update in range(num_updates):
            trajectories = collect_trajectories(env, agent, episodes_per_update)

        if trajectories:
            try:
                ppo_update(agent, trajectories, clip_param, ppo_epochs, batch_size)
                agent.step_schedulers()
            except Exception as e:
                print(f"Error in PPO update: {e}")
        else:
            print("No valid trajectories for update")
            continue

        # Evaluation with error handling and multiple trials
        try:
            s_eval, accuracy, model_size, computation_time, reward = evaluate_agent(env, agent, get_acc)

            with open(file_path, 'a') as f:
                f.write(f"{update},{s_eval:.4f},{accuracy:.4f},{model_size/(1024*1024):.4f},{computation_time:.6f},{reward:.4f}\n")

            print(f"Evaluation - Sparsity: {s_eval:.4f}, Accuracy: {accuracy:.4f}, Reward: {reward:.4f}")

            if accuracy >= MIN_ACCURACY and (
                (s_eval > best_sparsity and abs(accuracy - best_accuracy) < 0.5) or  # Better sparsity with similar accuracy
                reward > best_reward  # Better overall reward
            ):
                best_sparsity = s_eval
                best_accuracy = accuracy
                best_reward = reward
                best_model_size = model_size
                best_comp_time = computation_time
                no_improvement_count = 0  # Reset counter
                print(f"New best result found! Sparsity: {best_sparsity:.4f}, Accuracy: {best_accuracy:.4f}, Reward: {best_reward:.4f}")

                # Save the best model checkpoint
                save_best_model(agent, best_sparsity, best_accuracy, best_reward, best_model_size, best_comp_time)
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} updates")

            # Early stopping
            if no_improvement_count >= patience:
                print(f"Early stopping after {update+1} updates due to no improvement")
                break

        except Exception as e:
            print(f"Error in evaluation: {e}")

    return s_eval, accuracy, model_size, computation_time, best_sparsity, best_accuracy, best_model_size, best_comp_time, best_reward
        
def main():
    print(f"Minimum acceptable accuracy: {MIN_ACCURACY:.5f}")

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
        print("Best sparsity: {:.5f}".format(best_sparsity))
        print("Best accuracy: {:.5f}".format(best_accuracy))
        print("Best model size: {:.5f}MB".format(best_model_size/(1024*1024)))
        print("Best computation time: {:.5f}s".format(best_comp_time))
        print("Best reward: {:.6f}".format(best_reward))
        
        # Also show final results
        print("\n--- Final Results ---")
        print("Final sparsity: {:.5f}".format(s_eval))
        print("Final accuracy: {:.5f}".format(accuracy))
        print("Final model size: {:.5f}MB".format(model_size/(1024*1024)))
        print("Final computation time: {:.5f}s".format(computation_time))
    except Exception as e:
        print(f"Error in main training loop: {e}")
    
if __name__ == '__main__':
    get_min_acc()
    main()