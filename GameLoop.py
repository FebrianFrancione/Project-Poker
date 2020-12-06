from pypokerengine.api.game import setup_config, start_poker
from HonestPlayer import HonestPlayer
from RLPokerPlayer import RLPokerAgent
from BluffingPlayer import BluffingPlayer
import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(reward_list, subtitle):
    number_of_rounds = len(reward_list)
    rounds_list = np.arange(1, number_of_rounds + 1)
    plt.title('Training Reward over time ' + subtitle)
    plt.ylabel('Reward')
    plt.xlabel('Rounds')
    cumulative_rewards = np.cumsum(reward_list)
    plt.plot(rounds_list, cumulative_rewards)
    plt.tight_layout()
    plt.show()

def play_game(player1, player2, subtitle, number_rounds):

    total_rewards = []

    for i in range(number_rounds):
        print("********GAME LOOP #" + str(i) + "**********")
        config = setup_config(max_round=1, initial_stack=1000, small_blind_amount=10)
        config.register_player(name="p1", algorithm=player1)
        config.register_player(name="p2", algorithm=player2)
        game_result = start_poker(config, verbose=1)

        rewards_list_from_loop = player1.get_rewards_list()

        total_rewards.extend(rewards_list_from_loop)

        player1.clear_rewards()
        player1.clear_percentage_wins()

    plot_rewards(total_rewards, subtitle)

# declare our players
RLPokerPlayer = RLPokerAgent()
HonestPlayer = HonestPlayer()
RLPokerPlayer2 = RLPokerAgent()
BluffingPlayer = BluffingPlayer()

# play_game(RLPokerPlayer, HonestPlayer, "against Honest Player", 1000)
play_game(RLPokerPlayer, BluffingPlayer, "against Bluffing Player", 1000)
# play_game(RLPokerPlayer, RLPokerPlayer2, "against RL Player", 1000)
