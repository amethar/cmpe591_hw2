import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from hw2env import Hw2Env
from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import itertools

import os


# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

# Deep Q-Learning Agent
class Agent():

    def __init__(self, hyperparameter_set):

        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.gamma  			= hyperparameters['gamma']     				# discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.hidden_layer_dim   = hyperparameters['hidden_layer_dim']
        self.num_actions        = hyperparameters['num_actions']
        self.num_states         = hyperparameters['num_states']
        self.train_num_per_episode = hyperparameters['train_num_per_episode']        # at the end of each episode, call the train function this times


        # Neural Network
        #self.loss_fn = nn.MSELoss()         # NN Loss function, mean Squared Error 
        self.loss_fn = nn.SmoothL1Loss()     # NN Loss function, Huber loss 
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.PLOT_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')


    def train(self, render="offscreen"):

        start_time = datetime.now()
        last_plot_update_time = start_time
        last_network_save_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        # Create instance of the environment.
        env = Hw2Env(n_actions=self.num_actions, render_mode=render)


        # List to keep track of rewards collected per episode.
        rewards_per_episode = []
        steps_per_episode = []

        # Create policy network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(self.num_states, self.num_actions, self.hidden_layer_dim).to(device)

        # Create the target network and make it identical to the policy network
        target_dqn = DQN(self.num_states, self.num_actions, self.hidden_layer_dim).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # Initialize replay memory
        memory = ReplayMemory(self.replay_memory_size)


        # Initialize epsilon
        epsilon = self.epsilon_init

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track best reward
        best_reward = -9999999

        # Track number of trainings taken. Used for syncing policy and target network.
        trainings_from_last_update=0


        # Train INDEFINITELY, manually stop the run when you are satisfied  with the results
        for episode in itertools.count():

            env.reset()  # Initialize environment

            state = env.high_level_state()
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or reaches max step
            episode_reward = 0.0    # To accumulate rewards per episode
            episode_steps = 0

            # Perform actions until episode terminates
            while(not terminated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon: #select random action
                    action = np.random.randint(self.num_actions)
                else: # select best action
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()


                # Execute action
                new_state, reward, is_terminal, is_truncated = env.step(action)
                terminated = is_terminal or is_truncated

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                episode_steps+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)
            steps_per_episode.append(episode_steps)


            # Save model when new best reward is obtained.
            if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            # Save model every 600 seconds
            current_time = datetime.now()
            if current_time - last_network_save_time > timedelta(seconds=300):
                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                last_network_save_time = current_time


            # Update plot every 60 seconds
            current_time = datetime.now()
            if current_time - last_plot_update_time > timedelta(seconds=60):
                self.save_plot(rewards_per_episode, steps_per_episode, epsilon_history)
                last_plot_update_time = current_time

            # Train the network:
            if len(memory)>self.mini_batch_size:
                #Train this times:
                for train in range(self.train_num_per_episode):
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    # Increment tranings counter
                    trainings_from_last_update+=1

                # Copy policy network to target network after a certain number of trainings
                if trainings_from_last_update > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    trainings_from_last_update=0

                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)


    def test(self, render="gui"):

        # Create instance of the environment.
        env = Hw2Env(n_actions=self.num_actions, render_mode=render)


        # List to keep track of rewards collected per episode.
        rewards_per_episode = []
        steps_per_episode = []

        # Create policy network. 
        policy_dqn = DQN(self.num_states, self.num_actions, self.hidden_layer_dim).to(device)

        # Load learned policy
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

        # Switch model to evaluation mode
        policy_dqn.eval()


        # Test INDEFINITELY, manually stop the run when you are satisfied
        for episode in itertools.count():

            env.reset()  # Initialize environment
            state = env.high_level_state()

            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode
            episode_steps = 0

            # Perform actions until episode terminates
            while(not terminated):
  
                # Select the best action
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()

                # Execute action
                new_state, reward, is_terminal, is_truncated = env.step(action)
                terminated = is_terminal or is_truncated

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)


                # Increment step counter
                episode_steps+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)
            steps_per_episode.append(episode_steps)
            print(f"Episode={episode}, reward={episode_reward}, RPS={episode_reward/episode_steps}, steps={episode_steps}")


    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.gamma * target_dqn(new_states).max(dim=1)[0]

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients

        torch.nn.utils.clip_grad_value_(policy_dqn.parameters(), clip_value=50.0) # Apply gradient clipping

        self.optimizer.step()       # Update network parameters i.e. weights and biases



    def save_plot(self, rewards_per_episode, steps_per_episode, epsilon_history):
        # Save plots
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        # Plot raw rewards per episode
        axes[0].plot(rewards_per_episode, label="Cumulative Reward", color='blue')
        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].set_title('Cumulative Rewards per Episode')
        axes[0].legend()

        # Plot mean cumulative rewards per episode
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        axes[1].plot(mean_rewards, label="Smoothed Cumulative Reward", color='green')
        axes[1].set_xlabel('Episodes')
        axes[1].set_ylabel('Smoothed Cumulative Reward')
        axes[1].set_title('Smoothed Cumulative Rewards per Episode')
        axes[1].legend()

        # To plot raw rewards per step
        rewards_per_episode = np.array(rewards_per_episode)
        steps_per_episode = np.array(steps_per_episode)

        # Prevent division by zero (avoid errors when steps_per_episode contains zeros)
        steps_per_episode[steps_per_episode == 0] = 1  # Replace 0s with 1s to avoid division errors

        axes[2].plot(rewards_per_episode / steps_per_episode, label="Reward Per Step", color='blue')
        #axes[2].plot(rewards_per_episode/steps_per_episode, label="Reward Per Step", color='blue')
        axes[2].set_xlabel('Episodes')
        axes[2].set_ylabel('Reward Per Step')
        axes[2].set_title('Rewards per Step')
        axes[2].legend()

        # Plot epsilon decay
        axes[3].plot(epsilon_history, label="Epsilon Decay", color='red')
        axes[3].set_xlabel('Episodes')
        axes[3].set_ylabel('Epsilon')
        axes[3].set_title('Epsilon Decay')
        axes[3].legend()

        # Adjust layout and save plot
        plt.tight_layout()
        fig.savefig(self.PLOT_FILE)
        plt.close(fig)        



if __name__ == '__main__':

    dql = Agent(hyperparameter_set="parameters9")

    #dql.train()
    dql.test()
