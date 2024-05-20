import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F

# Function to convert the current state to a one-hot encoded tensor
def dqn_input(now_st,num_st):
    input_tensor = torch.zeros(num_st, dtype=torch.float32, device=device)
    input_tensor[now_st] = 1 
    return input_tensor

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DQN class
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes, dtype=torch.float32, device=device)  
        self.out = nn.Linear(h1_nodes, out_actions, dtype=torch.float32, device=device)

    def forward(self, x):
        # Define the forward pass through the network
        x = F.relu(self.fc1(x)) 
        x = self.out(x)       
        return x

# Initialize the environment
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
num_states = env.observation_space.n  # Number of states
num_actions = env.action_space.n  # Number of actions

# Load learned policy
policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))  # Load trained model weights
policy_dqn.eval()  # switch model to evaluation mode


# Run multiple episodes to test the learned policy
for i in range(10):
    now_state = env.reset()[0]  # Initialize to the starting state
    done = False  # Flag to indicate if the episode is finished

    # Agent navigates the map until it falls into a hole, reaches the goal, or takes too many actions
    while not done:
        # Select the best action using the policy network
        with torch.no_grad():
            action = policy_dqn(dqn_input(now_state, num_states)).argmax().item()

        # Execute the selected action
        new_state, reward, done, truncated, __ = env.step(action)
        now_state = new_state  # Update the current state

# Close the environment
env.close()