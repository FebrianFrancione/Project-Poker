from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator, DataEncoder
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Here we build the Neural Network to use for our Policy Network and Target Network
# 2 Fully Connected Hidden Layers
# 1 Output Layer
# We input the game state
# Output Layer has 4 possible outcomes [CALL, FOLD, RAISE_MIN, RAISE_MAX]

class DQN(nn.Module):
    # need img height and width
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=4)

    # Forward pass to network, esentially passing tensor t through the network, standard implementation
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


# Experience Class, here we define the Experience Object to be used for our Replay Memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


# The Class we use to store Experience Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    # add experience but check the memory isn't at capacity, else put the new memories in front
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    # what we use to train DQN, returns random sample of experiences
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # returns boolean to tell if we can sample from memory
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


# exploration = agent randomly exploring
# exploitation = agent exploits what it's learned to take best action
# epsilon greedy strategy to balance between both
# exploration becomes less probable the more the agent explores and learns about it's environment
class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    # The formula for Epsilon Greedy but in code
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)

    # strategy and number of available actions, implement this in RlPlayer


class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    # policy_net = deep Q network
    # rate is what we use from epsilon greedy
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        # pick action based on rate
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    return (t1,t2,t3,t4)

class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


# strategy and number of available actions
class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    # policy_net = deep Q network
    # rate is what we use from epsilon greedy
    def select_action(self, win_rate, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        # pick action based on rate
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(win_rate).argmax(dim=1).to(self.device)  # exploit

class RLPokerAgent(BasePokerPlayer):
    # hyperparameters and other setup for RL Player
    def __init__(self):
        self.batch_size = 256
        self.gamma = 0.999
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 0.001
        self.target_update = 10
        self.memory_size = 100000
        self.lr = 0.001
        self.episode_reward_amount = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)
        self.memory = ReplayMemory(self.memory_size)
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())


        # this network is not in training mode
        self.target_net.eval()

        # pass policy network
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        #storing the variable we need for the Q table
        self.state = 0
        self.next_state = 0
        self.last_action = []
        self.reward = 0

    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=1000,
            nb_player=2,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )

        agent = Agent(self.strategy, 4, self.device)
        action = agent.select_action(win_rate, self.policy_net)

        self.last_action = action
        self.state = self.next_state
        self.next_state = win_rate

        if action == 0:
            action_to_execute = valid_actions[0]
            print("************RL Player: Fold************")
            return action_to_execute['action'], action_to_execute['amount']
        elif action == 1:
            action_to_execute = valid_actions[1]
            print("************ RL Player: Call ************")
            return action_to_execute['action'], action_to_execute['amount']
        elif action == 2:
            action_to_execute = valid_actions[2]
            print("************ RL Player: Raise Min ************")
            print(action_to_execute['amount']['min'])
            print(action_to_execute['amount']['max'])
            return action_to_execute['action'], action_to_execute['amount']['min']
        elif action == 3:
            action_to_execute = valid_actions[2]
            print("************ RL Player: Raise Max ************")
            return action_to_execute['action'], action_to_execute['amount']['max']


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def get_stack(self, round_state):
        uuid = self.uuid
        seats = round_state['seats']
        stack_amount = 0
        for player in seats:
            if player['uuid'] == uuid:
                stack_amount = player['stack']
                break
        return stack_amount

    # This is where the round ends so we update our Experience and Update QValues here
    def receive_round_result_message(self, winners, hand_info, round_state):

        reward = self.get_stack(round_state)
        print(reward)
        action = self.last_action
        # Reward = Stack size
        state = self.state
        next_state = self.next_state
        self.memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if self.memory.can_provide_sample(self.batch_size):
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(self.policy_net, states, actions)
            next_q_values = QValues.get_next(self.target_net, next_states)
            target_q_values = (next_q_values * self.gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

