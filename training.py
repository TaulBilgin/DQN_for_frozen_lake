import gymnasium as gym
import random
import torch
from torch import nn
import torch.nn.functional as F

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to convert the state into a one-hot encoded tensor
def dqn_input(state, num_states):
    input_tensor = torch.zeros(num_states, dtype=torch.float32, device=device)
    input_tensor[state] = 1
    return input_tensor


# Function to optimize the policy network
def optimize(memory, policy_dqn, target_dqn, num_states, learning_rate, loss_fn, gamma):
    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=learning_rate)
    current_q_list = []
    target_q_list = []

    # Iterate over the memory to update the Q-values
    for now_state, action, new_state, reward, done in memory:
        # Calculate the target Q-value
        if reward != 0:
            target_value = reward
        else:
            target_value = gamma * target_dqn(dqn_input(new_state, num_states)).max().item()

        # Get the current Q-values from the policy network
        current_q = policy_dqn(dqn_input(now_state, num_states))
        # Clone the current Q-values to update the specific action value
        target_q = current_q.clone()
        target_q[action] = target_value

        current_q_list.append(current_q)
        target_q_list.append(target_q)
    
    # Compute the loss between current Q-values and target Q-values
    loss = loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

    # Perform backpropagation and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update the target network to match the policy network
    target_dqn.load_state_dict(policy_dqn.state_dict())

    

# Define the DQN class
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        # Define the forward pass through the network
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


def training_loop():
    run = 0  # Counter for the number of episodes
    learning_rate = 0.01  # Learning rate for the optimizer
    gamma = 0.9  # Discount factor for future rewards
    choice_list = ['x'] * 100  # List for epsilon-greedy action selection
    total_rewards = 0  # Counter for total rewards
    previous_total_rewards = 0  # Counter for total rewards in the previous iteration
    loss_fn = nn.MSELoss()  # Mean Squared Error loss function
    memory = []  # Memory buffer for storing experiences

    # Initialize the environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize policy and target networks
    policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions).to(device)
    target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions).to(device)
    target_dqn.load_state_dict(policy_dqn.state_dict()) # Copy weights from policy to target network

    while True :
        now_state = env.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished

        run += 1 # Increment the episode counter
        # Play one episode
        while not done:
            # Epsilon-greedy action selection
            if random.choice(choice_list) == "x":
                action = env.action_space.sample()  # Random action
            else:
                with torch.no_grad():
                    action = policy_dqn(dqn_input(now_state, num_states)).argmax().item()  # Best action

            # Take action and observe result
            new_state, reward, done, truncated, _ = env.step(action)

            if reward == 1:
                total_rewards += 1

            # Store transition in memory
            memory.append((now_state, action, new_state, reward, done))
            now_state = new_state

        print(f"run : {run} | totally_accurate : {total_rewards}")

        # Stop training if the maximum number of episodes is reached
        if run > 5000:
            env.close()
            return policy_dqn
        
        # Perform optimization if memory size is sufficient
        if len(memory) > 32:
            optimize(memory, policy_dqn, target_dqn, num_states, learning_rate, loss_fn, gamma)
            if previous_total_rewards != total_rewards:
                # Check if the policy has improved
                if all(choice == 'y' for choice in choice_list):
                    env.close()
                    return policy_dqn

                choice_list.remove("x")
                choice_list.append("y")
                previous_total_rewards = total_rewards

            memory = [] # Clear memory after optimization

# Function to test the trained policy network
def test_for_save(policy_dqn):
    env2 = gym.make("FrozenLake-v1", is_slippery=False)
    num_states = env2.observation_space.n
    now_state = env2.reset()[0]
    done = False

    # Play one episode with the trained policy
    while not done:
        with torch.no_grad():
            action = policy_dqn(dqn_input(now_state, num_states)).argmax().item()
        new_state, reward, done, truncated, _ = env2.step(action)
        now_state = new_state

    return reward


# Main loop to train and save the policy network
while True:
    policy_dqn = training_loop()
    success = test_for_save(policy_dqn)
    if success == 1:
        torch.save(policy_dqn.state_dict(), "frozen_lake_dql.pt")
        print("policy_dqn is saved")
        break
    else:
        print("policy_dqn is not saved, retraining")
