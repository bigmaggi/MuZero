import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from torch import optim
import random

# Set up device
device = torch.device("mps")

env = gym.make("CartPole-v1", render_mode="human")

# Define the Representation Function
class RepresentationFunction(nn.Module):
    def __init__(self, representation_size, state_size, action_size):
        super(RepresentationFunction, self).__init__()
        self.representation_size = representation_size
        self.state_size = state_size
        self.action_size = action_size
        self.representation_model = nn.Linear(state_size + action_size, representation_size).to(device)

    def forward(self, state, action):
        action = action.unsqueeze(-1)  # add an extra dimension to make action a 2D tensor
        state_repr = self.representation_model(torch.cat([state, action], dim=-1))
        return state_repr


# Define the Initial Representation Function
class InitialRepresentationFunction(nn.Module):
    def __init__(self, input_size, representation_size):
        super(InitialRepresentationFunction, self).__init__()
        self.representation_size = representation_size
        self.initial_representation_model = nn.Linear(input_size, representation_size).to(device)

    def forward(self, state):
        initial_state_repr = self.initial_representation_model(state)
        initial_state_repr = initial_state_repr.unsqueeze(0)
        return initial_state_repr

# Define the Dynamics Function
class DynamicsFunction(nn.Module):
    def __init__(self):
        super(DynamicsFunction, self).__init__()
        self.fc1 = nn.Linear(64 + 1, 128).to(device)
        self.fc2 = nn.Linear(128, 128).to(device)
        self.reward_output = nn.Linear(128, 1).to(device)  # reward prediction
        self.state_output = nn.Linear(128, 64).to(device)  # next state prediction

    def forward(self, state_repr, action):
        action_tensor = action.unsqueeze(-1).to(device)  # Convert action to a tensor
        x = torch.cat([state_repr, action_tensor], dim=-1)
        x = F.relu(self.fc1(x))
        # print(f"Shape of state_repr in DynamicsFunction: {state_repr.shape}")
        next_state_repr = self.state_output(x)
        # print(f"Shape of next_state_repr in DynamicsFunction: {next_state_repr.shape}")
        next_rewards = self.reward_output(x)
        return next_state_repr, next_rewards

# Define the Prediction Function
class PredictionFunction(nn.Module):
    def __init__(self):
        super(PredictionFunction, self).__init__()
        self.fc1 = nn.Linear(64, 128).to(device)
        self.policy_output = nn.Linear(128, 2).to(device)  # policy output (2 actions in CartPole-v1)
        self.value_output = nn.Linear(128, 1).to(device)  # value output

    def forward(self, hidden_state):
        x = F.relu(self.fc1(hidden_state))
        return F.softmax(self.policy_output(x), dim=1), self.value_output(x)  # return both the policy and the value

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
        self.num_simulations = 50
        self.discount = 0.99
        self.root = None
        self.initial_representation_function = initial_representation_function  # new line
        self.representation_function = representation_function
        self.dynamics_function = dynamics_function
        self.prediction_function = prediction_function

    def UCB_score(self, parent, action):
        """
        The UCB formula involves the visit counts of the parent and action,
        the total value of the action, and an exploration constant.
        """
        UCB_C = 2.0  # Increased exploration constant from 1.0 to 2.0
        total_visits = sum(parent.visit_count)
        return parent.total_value[action] / parent.visit_count[action] + UCB_C * math.sqrt(math.log(total_visits) / parent.visit_count[action])


    def expand(self, node, action):
        next_state, reward = self.dynamics_function(node.hidden_state, torch.tensor([action], dtype=torch.float32).to(device))
        next_state = next_state.clone().detach().to(device)
        reward = reward.item()
        return Node(next_state, reward, False, self.action_space)

    def backpropagate(self, leaf_value, path):
        """
        Path is a list of nodes from the root to the leaf.
        We reverse it so that we're going from the leaf to the root.
        """
        for node, action in reversed(path):
            node.visit_count[action] += 1
            node.total_value[action] += leaf_value

    def run(self, initial_state):
        # Create root node with initial state
        initial_hidden_state = self.initial_representation_function(initial_state)
        self.root = Node(initial_hidden_state, 0, False, self.action_space)

        for _ in range(self.num_simulations):
            node, path = self.root, []
            while True:
                action = max(range(self.action_space), key=lambda a: self.UCB_score(node, a) if node.children[a] is not None else float("-inf"))
                path.append((node, action))
                if node.children[action] is None:
                    break
                node = node.children[action]

            # The following two lines are replacing the original 'if node is None:' block.
            # We expand the leaf node (node.children[action] is None before the expansion), and retrieve the reward.
            leaf = self.expand(node, action)
            leaf_value = leaf.reward

            # Backpropagation step (update all nodes in path)
            self.backpropagate(leaf_value, path)

        # return the action with the highest visit count at the root node
        return max(range(self.action_space), key=lambda a: self.root.visit_count[a])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Create the MCTS object
action_space = env.action_space.n  # assuming env is a Gym environment
representation_size = 64  # Replace 64 with the desired size for the state representation

# Create instances of the Representation, Dynamics, and Prediction Functions
dynamics_function = DynamicsFunction()
prediction_function = PredictionFunction()

state_size = env.observation_space.shape[0]  # assuming state has 4 features
action_size = 1  # assuming action has 1 feature
representation_function = RepresentationFunction(representation_size, state_size, action_size)
target_representation_function = RepresentationFunction(representation_size, state_size, action_size)
initial_representation_function = InitialRepresentationFunction(env.observation_space.shape[0], representation_size)

target_dynamics_function = DynamicsFunction()
target_prediction_function = PredictionFunction()

# Copy weights from the networks to the target networks
target_representation_function.load_state_dict(representation_function.state_dict())
target_dynamics_function.load_state_dict(dynamics_function.state_dict())
target_prediction_function.load_state_dict(prediction_function.state_dict())

gamma = 0.99  # Discount factor for the Bellman equation

mcts = MCTS(action_space, initial_representation_function, representation_function, dynamics_function, prediction_function)

def run_simulation(env, mcts, replay_buffer):
    # Create initial state
    state = env.reset()
    if isinstance(state, tuple) and len(state) == 2:
        state, _ = state  # Extract the state from the tuple
    state = torch.from_numpy(state).float().to(device)

    done = False
    total_reward = 0
    while not done:
        # Get action from MCTS
        action = mcts.run(state)

        # Execute action in environment
        next_state, reward, done, _, _ = env.step(action)
        env.render()  # <-- Add this line to visualize the environment

        next_state = torch.from_numpy(next_state).float().to(device)

        total_reward += reward

        # Store experience in replay buffer
        replay_buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

        # Update current state
        state = next_state

    return total_reward


def compute_loss(batch):
    """
    Compute the loss for the given batch of data.
    """
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)

    # Compute predictions with networks
    # print(f"Shape of states in compute_loss: {states.shape}")
    state_repr = representation_function(states, actions)
    # print(f"Shape of state_repr in compute_loss: {state_repr.shape}")
    next_state_repr, rewards_pred = dynamics_function(state_repr, actions)
    # print(f"Shape of next_state_repr in compute_loss: {next_state_repr.shape}")
    next_state_repr_target = target_representation_function(next_states, actions)
    # print(f"Shape of next_state_repr_target in compute_loss: {next_state_repr_target.shape}")

    policies_pred, values_pred = prediction_function(state_repr)

    # Compute targets with target networks
    next_state_repr_target, _ = target_dynamics_function(target_representation_function(next_states, actions), actions)
    # print("Shape of next_state_repr_target:", next_state_repr_target.shape)
    _, next_values_target = target_prediction_function(next_state_repr_target)

    expected_rewards = rewards + gamma * next_values_target * (1 - dones)

    # Compute loss components
    prediction_loss = F.mse_loss(rewards_pred, rewards)
    value_loss = F.mse_loss(values_pred, expected_rewards.detach())
    policy_loss = F.cross_entropy(policies_pred, actions)

    return prediction_loss + value_loss + policy_loss

# Create optimizers
optimizer_repr = optim.Adam(representation_function.parameters())
optimizer_dyn = optim.Adam(dynamics_function.parameters())
optimizer_pred = optim.Adam(prediction_function.parameters())

# Now we can just call run_simulation function to start the simulation
replay_buffer = ReplayBuffer(capacity=10000)

loss = 0

# Run the training loop
for episode in range(10000):
    total_reward = run_simulation(env, mcts, replay_buffer)

    if len(replay_buffer) > 32:
        # Sample batch from replay buffer
        batch = replay_buffer.sample(32)

        # Compute loss
        loss = compute_loss(batch)

        # Backward pass and optimization
        optimizer_repr.zero_grad()
        optimizer_dyn.zero_grad()
        optimizer_pred.zero_grad()
        loss.backward()
        optimizer_repr.step()
        optimizer_dyn.step()
        optimizer_pred.step()

        # Update target networks
        if episode % 100 == 0:  # every 100 episodes
            target_representation_function.load_state_dict(representation_function.state_dict())
            target_dynamics_function.load_state_dict(dynamics_function.state_dict())
            target_prediction_function.load_state_dict(prediction_function.state_dict())

        print("Episode: {}, Total Reward: {}, Loss: {}".format(episode, total_reward, loss.item()))
    else:
        print("Episode: {}, Total Reward: {}".format(episode, total_reward))
