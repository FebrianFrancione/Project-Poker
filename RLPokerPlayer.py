from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import math
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Here we build the Neural Network to use for our Policy Network and Target Network
# 2 Fully Connected Hidden Layers
# 1 Output Layer
# We input the game state
# Output Layer has 4 possible outcomes [CALL, FOLD, RAISE_MIN, RAISE_MAX]

class DQN(nn.Module):
    # need img height and width
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=256, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=4)

    # Forward pass to network, esentially passing tensor t through the network, standard implementation
    def forward(self, t):
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
    # it will decay over time, meaning it explores initially then exploits its environment
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
    # this is used to sample the memory to use for policy network
    batch = Experience(*zip(*experiences))
    t1 = torch.tensor(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.tensor(batch.reward)
    t4 = torch.tensor(batch.next_state)
    return (t1,t2,t3,t4)

class QValues:
    device = torch.device("cpu")

    # state action pairs from replay memory, return predicted Q value
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=0, index=actions)

    # For each item in next state we want to obtain the max Q value predicted by the target net
    # among all possible next actions
    @staticmethod
    def get_next(target_net, next_states):
        final_states_locations = next_states.type(torch.bool)
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[final_states_locations] = target_net(next_states).max()
        return values

class RLPokerAgent(BasePokerPlayer):
    # hyper parameters and other setup for RL Player
    # The only hyper parameter modified from the example was the memory size
    # this is because 1000 is very small to represent all the possible states we could have in poker
    # Increasing this technically will decrease performance but a large drop in performance was not observed
    def __init__(self):
        self.batch_size = 256
        self.gamma = 0.999
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 0.001
        self.target_update = 10
        self.memory_size = 100000
        self.lr = 0.001

        # here we declare our networks, strategy and memory as part of the RL class
        self.device = torch.device("cpu")
        self.strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)
        self.memory = ReplayMemory(self.memory_size)
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # this network is not in training mode
        self.target_net.eval()

        # pass policy network
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        # storing the variable we need for the Q table
        # this will help us track the current state and next state
        # other variables will be used by class methods to be passed in main game loop for plotting
        self.state = 0
        self.next_state = 0
        self.last_action = []
        self.reward = 0
        self.last_stack_size = 0

        # other variables to make plots later
        self.round_number = 0
        self.round_reward_amount = []
        self.num_wins = 0
        self.list_percent_wins = []

    # this method is where the agent declares its action between fold, call, raise min, raise max
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=1000,
            nb_player=2,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card)
        )

        # here we declare the stack size BEFORE the agent takes it's action
        # we will subtract it from the stack size after to determine the reward for the round
        self.last_stack_size = self.get_stack(round_state)

        agent = Agent(self.strategy, 4, self.device)

        # we pass the win rate as the input to the network
        action = agent.select_action(win_rate, self.policy_net)

        # updating the state
        # since we are at a new decision point, the current win rate becomes the next state
        # and the current state becomes the win rate at the previous decision point
        self.last_action = action
        self.state = self.next_state
        self.next_state = win_rate

        # declaring which action to take
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

    # get the amount of chips in the stack at the moment this method is called
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
        # Reward = Stack size - stack size before action
        reward = self.get_stack(round_state) - self.last_stack_size
        action = self.last_action
        state = self.state
        next_state = self.next_state
        self.memory.push(Experience(state, action, next_state, reward))

        # adding the number of wins, here we assume if the reward is positive then the round is won
        if reward > 0:
            self.num_wins = self.num_wins + 1

        # add reward and round number, to be used for plotting
        self.round_number = self.round_number + 1
        self.round_reward_amount.append(reward)

        # store win rate for plotting
        number_rounds = len(self.round_reward_amount)
        percent_win_round = 100 * (self.num_wins / number_rounds)
        self.list_percent_wins.append(percent_win_round)

        # This is where we update our
        if self.memory.can_provide_sample(self.batch_size):
            experiences = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(self.policy_net, states, actions)
            next_q_values = QValues.get_next(self.target_net, next_states)
            target_q_values = (next_q_values * self.gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # this method will help us pass the information to the game loop so we can plot the information
    def get_rewards_list(self):
        return self.round_reward_amount

    # reset reward list
    def clear_rewards(self):
        self.round_reward_amount = []

    # to be used for plotting
    def get_percentage_list(self):
        return self.list_percent_wins, self.num_wins

    # clear percentage wins list and number of wins
    def clear_percentage_wins(self):
        self.list_percent_wins = []
        self.num_wins = 0

# this is needed to use the poker GUI
def setup_ai():
    return RLPokerAgent()
