import gymnasium as gym
import torch
from torch import nn
import math
import numpy as np
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import copy


# Set up device
# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RepresentationFunction(nn.Module):
    def __init__(self, state_size, representation_size=4):  # set default representation size to 4
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128).to(device)
        self.fc2 = nn.Linear(128, representation_size).to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x


# Define the Initial Representation Function
class InitialRepresentationFunction(nn.Module):
    def __init__(self, input_size, representation_size):
        super(InitialRepresentationFunction, self).__init__()
        self.representation_size = representation_size
        self.initial_representation_model = nn.Sequential(
            nn.Linear(input_size, 2 * representation_size),
            nn.LeakyReLU(),
            nn.Linear(2 * representation_size, 64),  
        ).to(device)

    def forward(self, state):
        initial_state_repr = self.initial_representation_model(state)
        return initial_state_repr


# Define the Dynamics Function
class DynamicsFunction(nn.Module):
    def __init__(self):
        super(DynamicsFunction, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(64 + 1, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
        ).to(device)
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(),
        ).to(device)
        self.reward_output = nn.Linear(128, 1).to(device)
        self.state_output = nn.Linear(128, 64).to(device)

    def forward(self, state_repr, action):
        action_tensor = action.float().unsqueeze(-1).to(device)
        x = self.fc1(torch.cat([state_repr, action_tensor], dim=-1))
        x = self.fc2(x)
        next_state_repr = self.state_output(x)
        next_rewards = self.reward_output(x)
        return next_state_repr, next_rewards



class PredictionFunction(nn.Module):
    def __init__(self, action_space_size):
        super(PredictionFunction, self).__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(64, 256),  # Change the input size to 64
            nn.ReLU(),
        ).to(device)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
        ).to(device)
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
        ).to(device)
        self.policy_head = nn.Linear(128, action_space_size).to(device)
        self.value_head = nn.Linear(128, 1).to(device)

    def forward(self, hidden_state):
        hidden_state = hidden_state.to(device)  # added this line
        print("Shape of hidden_state: ", hidden_state.shape)
        x = self.fc0(hidden_state)  # added this line
        print("Shape after fc0: ", x.shape)
        x = self.fc1(x)
        print("Shape after fc1: ", x.shape)
        x = self.fc2(x)
        print("Shape after fc2: ", x.shape)
        print(f'Shape of x: {x.shape}')
        policy_output = self.policy_head(x)
        print(f'Shape of policy_output: {policy_output.shape}')
        return F.softmax(policy_output, dim=0), self.value_head(x)  # return both the policy and the value

class Config:
    def __init__(self):
        self.environment_name = "LunarLander-v2"
        self.seed = 42069
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.training_steps = 10000
        self.lr = 0.001
        self.min_priority = 0.1
        self.checkpoint_interval = 1000
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.discount = 0.997
        self.exploration_constant = 1.0
        self.learning_rate = 0.001
        self.representation_size = 64
        self.batch_size = 128
        self.num_epochs = 10
        #self.state_size = None
        #self.action_size = None
        self.replay_buffer_capacity = 10000
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.eps = 0.01
        self.num_simulations = 50
        self.temperature = 1.0
        self.dirichlet_alpha = 0.25
        self.noise_weight = 0.25
        self.gradient_clip = 40.0
        self.max_moves = 27000
        self.mcts_discount = self.discount
        self.episodes = 1000
        self.gamma = 0  # Discount factor for the Bellman equation
        self.lr_repr = 0.0001
        self.lr_dyn = 0.0001
        self.lr_pred = 0.0001
        self.update_interval = 10
        self.max_steps = 200
config = Config()


# Define the Monte Carlo Tree Search
# start with the node implementation
class Node:
    def __init__(self, hidden_state, reward, terminal, action_space):
        self.hidden_state = hidden_state
        self.reward = reward
        self.terminal = terminal
        self.children = [None] * action_space
        self.total_value = [0] * action_space
        self.visit_count = [0] * action_space

# then the MCTS implementation
class MCTS:
    def __init__(self, action_space, initial_representation_function, representation_function, dynamics_function, prediction_function):
        self.action_space = action_space
        self.num_simulations = config.num_simulations
        self.discount = config.mcts_discount
        self.root = None
        self.initial_representation_function = self._create_initial_representation_function()
        self.dynamics_function = dynamics_function
        self.prediction_function = prediction_function
        self.exploration_constant = config.exploration_constant

    def UCB_score(self, node, action):
        """
        Compute the UCB score for a child node.
        """
        if node.visit_count[action] == 0:
            return float('inf')
        else:
            # Use the model to predict the value of the action
            state = self.representation_function(node.hidden_state)
            predicted_values = self.prediction_function(state)
            Q = predicted_values[action]
            U = self.exploration_constant * math.sqrt(math.log(node.visit_count_sum) / node.visit_count[action])
            return Q + U

    def expand(self, node, action):
        next_state, reward = self.dynamics_function(node.hidden_state, torch.tensor([action], dtype=torch.float32).to(device))
        next_state = next_state.clone().detach().to(device)
        reward = reward.item()
        return Node(next_state, reward, False, self.action_space)

    def backpropagate(self, leaf_value, path):
        for node, action in reversed(path):
            node.visit_count += 1
            node.total_value += leaf_value
            leaf_value *= self.discount


    def run(self, initial_state):
        # Create root node with initial state
        initial_hidden_state = self.initial_representation_function(initial_state)
        self.root = Node(initial_hidden_state, 0, False, self.action_space)

        for _ in range(self.num_simulations):
            node, path = self.root, []
            while True:
                # Check if all child nodes are explored
                if all(child is None for child in node.children):
                    # use best action from the prediction function
                    action_probs, _ = self.prediction_function(self.representation_function(node.hidden_state))
                    action = torch.argmax(action_probs).item()
                else:
                    action = max(range(self.action_space),
                                 key=lambda a: self.UCB_score(node, a) if node.children[a] is not None else float("-inf"))

                path.append((node, action))
                if node.children[action] is None:
                    break
                node = node.children[action]

            # The following two lines are replacing the original 'if node is None:' block.
            # We expand the leaf node (node.children[action] is None before the expansion), and retrieve the reward.
            leaf = self.expand(node, action)
            leaf_value = leaf.reward

            self.backpropagate(leaf_value, path)
           

        # return the action with the highest visit count at the root node
        return max(range(self.action_space), key=lambda a: self.root.visit_count[a])

class ReplayBuffer:
    """
    This class implements the replay buffer.
    a transition is a tuple of (state, action, reward, next_state, done) which represents the agent's experience
    """
    def __init__(self, capacity):
        self.capacity = capacity  # maximum number of elements in the buffer
        self.buffer = []  # buffer to store transitions
        self.position = 0  # position to insert new transition

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:  # if buffer is not full, append new transition
            self.buffer.append(None)  # append None to buffer
        # Convert state and next_state to numpy if they are torch tensors
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        # Convert reward to float
        reward = float(reward)
        self.buffer[self.position] = (state, action, reward, next_state, done)  # insert new transition
        self.position = (self.position + 1) % self.capacity  # update position

    def sample(self, batch_size):
        # TODO: prioritied replay sampling
        batch = random.sample(self.buffer, batch_size) 
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.capacity = capacity  # maximum number of elements in the buffer
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # small constant to avoid zero priority
        self.buffer = []  # buffer to store transitions
        self.position = 0  # position to insert new transition
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # priorities array
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:  # if buffer is not full, append new transition
            self.buffer.append(None)  # append None to buffer
        else:
            self.priorities[self.position % self.capacity] = self.max_priority  # replace the oldest priority with the max priority

        self.buffer[self.position] = (state, action, reward, next_state, done)  # insert new transition
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity  # update position

    def sample(self, batch_size):
        total_priority = np.sum(self.priorities)
        probabilities = self.priorities / total_priority  # Normalize the priorities to get probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = 1 / (len(self.buffer) * probabilities[indices])
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return states, actions, rewards, next_states, dones, indices, weights  # Include indices in the returned value

    def update_priorities(self, indices, td_errors):
        td_errors = td_errors.flatten()
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + self.epsilon
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, env, config, input_size, representation_size, prediction_function):
        self.env = env
        self.config = config
        self.input_size = input_size
        self.representation_function = representation_function
        self.dynamics_function = dynamics_function
        self.prediction_function = prediction_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize target networks
        self.target_representation_function = representation_function.clone()
        self.target_dynamics_function = dynamics_function.clone()
        self.target_prediction_function = prediction_function.clone()

        # Set target networks in evaluation mode (no gradients)
        self.target_representation_function.eval()
        self.target_dynamics_function.eval()
        self.target_prediction_function.eval()

        # Disable gradient computation for target networks
        for param in self.target_representation_function.parameters():
            param.requires_grad = False
        for param in self.target_dynamics_function.parameters():
            param.requires_grad = False
        for param in self.target_prediction_function.parameters():
            param.requires_grad = False

        # Set the optimizer for the representation, dynamics, and prediction functions
        self.optimizer = optim.Adam(
            list(self.representation_function.parameters()) +
            list(self.dynamics_function.parameters()) +
            list(self.prediction_function.parameters()),
            lr=self.config.lr
        )

        # Create a replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(self.config.replay_buffer_size)

        # Initialize other variables
        self.best_reward = float("-inf")
        self.gamma = self.config.gamma


    def _create_action_space(self):
        # Check if the environment has a discrete action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        # Check if the environment has a continuous action space
        elif isinstance(self.env.action_space, gym.spaces.Box):
            return self.env.action_space.shape[0]
        else:
            raise ValueError("Unsupported action space type.")

    def _get_num_actions(self):
        # Check if the environment has a discrete action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            return self.env.action_space.n
        # Check if the environment has a continuous action space
        elif isinstance(self.env.action_space, gym.spaces.Box):
            return -1  # Indicate that it is a continuous action space
        else:
            raise ValueError("Unsupported action space type.")

    def _get_state_size(self):
        # Check if the environment has an observation space
        if hasattr(self.env, "observation_space"):
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                return self.env.observation_space.n
            elif isinstance(self.env.observation_space, gym.spaces.Box):
                return self.env.observation_space.shape[0]
        raise ValueError("The environment does not have an observation space.")

    def _create_representation_function(self, input_size, representation_size):
        # Define a simple fully connected neural network for representation
        return nn.Linear(input_size, representation_size)

    def _create_initial_representation_function(self, state_size):
        # Your implementation of creating the initial representation function goes here
        # Make sure to use the self.representation_function instead of self.representation_size
        representation_function = ...  # Your code to create the representation function
        return representation_function

    def _create_dynamics_function(self):
        return DynamicsFunction()

    def _create_prediction_function(self, action_space_size):
        return PredictionFunction(action_space_size)


    def _create_replay_buffer(self, capacity):
        return ReplayBuffer(capacity)

    def _create_prioritized_replay_buffer(self, capacity):
        return PrioritizedReplayBuffer(capacity, alpha=config.alpha, beta=config.beta, beta_increment=config.beta_increment, epsilon=config.eps)

    def _create_mcts(self):
        return MCTS(self.action_space, self.initial_representation_function,
                    self.representation_function, self.dynamics_function, self.prediction_function,
                    num_simulations=config.num_simulations, discount=config.mcts_discount, exploration_constant=config.exploration_constant)

    def _create_optimizer(self):
        params = list(self.representation_function.parameters()) + \
                  list(self.dynamics_function.parameters()) + \
                  list(self.prediction_function.parameters())
        return torch.optim.Adam(params, lr=self.config.learning_rate)


    def run_simulation(agent, initial_state):
        state = torch.tensor(initial_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        done = False
        total_reward = 0
        step_count = 0
        max_steps = agent.config.max_steps

        # Get the initial state representation
        state_repr = agent.representation_function(state)

        while not done and step_count < max_steps:
            step_count += 1

            # Get action probabilities and values from prediction function
            policies_pred, _ = agent.prediction_function(state_repr)

            # Choose action using epsilon-greedy policy
            epsilon = agent.config.exploration_constant
            if random.random() < epsilon:
                action = random.randint(0, agent.action_space - 1)
            else:
                action = torch.argmax(policies_pred).item()

            # Take a step in the environment
            next_state, reward, done, _ = agent.env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            total_reward += reward

            # Get the next state representation
            next_state_repr = agent.representation_function(next_state)

            # Store the transition in the replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Update current state representation
            state_repr = next_state_repr

        return total_reward



    def compute_loss(self, states, actions, rewards, next_states, dones, weights):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Compute TD targets
        with torch.no_grad():
            next_state_repr = self.target_representation_function(next_states)
            _, next_state_values = self.target_prediction_function(next_state_repr)
            td_targets = rewards + self.gamma * (1 - dones) * next_state_values

        # Compute TD errors and loss
        state_repr = self.representation_function(states)
        rewards_pred, values_pred = self.prediction_function(state_repr)
        state_values = values_pred.gather(1, actions)
        td_errors = td_targets - state_values.squeeze()
        loss = (weights * F.smooth_l1_loss(rewards_pred, td_targets.detach(), reduction='none')).mean()

        return loss

    # Reward Function
    def compute_reward_cartpole(self, next_state):
    # def compute_reward(self, next_state):
        x, x_dot, theta, theta_dot = next_state

        # reward for every timestep it stays alive
        reward = 1.0

        # reward for being near the center
        reward -= 0.01 * (x**2)  

        # penalty for large pole angle
        reward -= 0.01 * (theta**2)  

        # penalty for large cart speed
        reward -= 0.001 * (x_dot**2)

        # penalty for large pole angular speed
        reward -= 0.001 * (theta_dot**2)

        # bonus reward for small pole angle and small cart movement
        if abs(x) < 0.05 and abs(theta) < 0.05:
            reward += 1.0

        return torch.tensor([reward], device=device)


    def compute_reward_mountain_car(self, next_state):
        velocity, position = next_state

        reward = 1.0
        # reward high positions
        reward += 0.01 * (position**2)
        # reward high velocities
        reward += 0.01 * (velocity**2)

        return torch.tensor([reward], device=device)

    
    def compute_reward_acrobot(self, next_state):
        cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta_dot1, theta_dot2 = next_state

        theta1 = math.acos(cos_theta1) * math.copysign(1, sin_theta1)
        theta2 = math.acos(cos_theta2) * math.copysign(1, sin_theta2)

        x1 = -0.5 * 2.0 * math.cos(theta1)
        y1 = -0.5 * 2.0 * math.sin(theta1)
        x2 = x1 - 1.0 * math.cos(theta1 + theta2)
        y2 = y1 - 1.0 * math.sin(theta1 + theta2)

        # The -0.5 is to subtract the length of the first link
        height = y2 + 2.0 + 0.5

        # Energy used by the system

        # Reward is maximum when height is maximum 
        reward = height

        # Penalize large angles
        reward -= 0.01 * (theta1**2 + theta2**2)
        # Penalize large angular velocities
        reward -= 0.01 * (theta_dot1**2 + theta_dot2**2)
        # Reward for being upright
        reward += 2.0 * (abs(theta1) < 0.05 and abs(theta2) < 0.05)
        # Reward for reaching the top
        reward += 2.0 * (height > 2.0)
        # Reward for reaching the top fast
        reward += 2.0 * (height > 2.0 and abs(theta_dot1) < 0.05 and abs(theta_dot2) < 0.05)
        # 

        return torch.tensor([reward], device=device)
    
    def compute_reward_lunar_lander(self, next_state):
        x, y, vx, vy, theta, vtheta, contact_left, contact_right = next_state.squeeze(0).cpu().numpy()
        reward = 0
        reward += 100 * np.sqrt(max(0, 1 - np.sqrt(x ** 2 + y ** 2)))
        reward -= 100 * np.sqrt(vx ** 2 + vy ** 2)
        reward -= 100 * abs(theta)
        reward -= 0.3 * vtheta ** 2
        if contact_left or contact_right:  # legs have contact
            reward += 10
        return torch.tensor([reward], device=device)

    def train(self, episodes=config.episodes):
        pbar = tqdm(range(episodes), desc="Episodes")
        running_reward = 0
        running_rewards = []
        losses = []

        for episode in pbar:
            total_reward = self.run_simulation()
            running_reward = 0.05 * total_reward + (1 - 0.05) * running_reward
            running_rewards.append(running_reward)

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                for model_name, model in {
                    "representation": self.representation_function,
                    "dynamics": self.dynamics_function,
                    "prediction": self.prediction_function
                }.items():
                    torch.save(model.state_dict(), self.model_paths[model_name])

            if len(self.replay_buffer) >= self.replay_buffer.capacity:
                states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.config.batch_size)
                td_errors = self.compute_td_errors(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
                loss = self.compute_loss(states, actions, rewards, next_states, dones, weights)
                losses.append(loss.item())

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update priorities in the replay buffer based on TD errors
                with torch.no_grad():
                    next_state_repr = self.representation_function(next_states)
                    _, next_state_values = self.target_prediction_function(next_state_repr)
                    target_values = rewards + self.gamma * (1 - dones) * next_state_values
                    state_repr = self.representation_function(states)
                    _, state_values = self.prediction_function(state_repr)
                    td_errors = torch.abs(target_values - state_values).cpu().numpy()
                    self.replay_buffer.update_priorities(indices, td_errors)

                # Log the training statistics
                print(f"Episode: {episode}, Loss: {loss.item()}, Reward: {total_reward}, Best Reward: {self.best_reward}")

            pbar.set_postfix({"Reward": total_reward, "Best Reward": self.best_reward})

        plt.plot(torch.tensor(running_rewards).cpu().numpy())
        plt.title('Running Average of Rewards Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Running Average of Reward')
        plt.show()

        plt.plot(torch.tensor(losses).cpu().numpy())
        plt.title('Loss Over Time')
        plt.xlabel('Episode')
        plt.ylabel('LossCompleting the `train_representation` function: ')


    def load_model(self):
        for model_name, model in {
            "representation": self.representation_function,
            "dynamics": self.dynamics_function,
            "prediction": self.prediction_function
        }.items():
            if os.path.isfile(self.model_paths[model_name]):
                model.load_state_dict(torch.load(self.model_paths[model_name]))
                print(f"Loaded {model_name} model parameters from disk.")
            else:
                print(f"No saved {model_name} model parameters found.")
    

    def compute_td_errors(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        next_state_repr = self.target_representation_function(next_states)
        _, next_state_values = self.target_prediction_function(next_state_repr)
        target_values = rewards + self.gamma * (1 - dones) * next_state_values
        state_repr = self.representation_function(states)
        _, state_values = self.prediction_function(state_repr)
        td_errors = torch.abs(target_values.detach() - state_values).detach().cpu().numpy()
        return td_errors



if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    input_size = env.observation_space.shape[0]
    representation_size = 64
    prediction_function = PredictionFunction(env.action_space.n)
    config = Config()
    agent = Agent(env, config, input_size, representation_size=representation_size, prediction_function=prediction_function)
    agent.train()