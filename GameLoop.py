from pypokerengine.api.game import setup_config, start_poker
from HonestPlayer import HonestPlayer
from RLPokerPlayer import RLPokerAgent

config = setup_config(max_round=20, initial_stack=100000, small_blind_amount=5)
config.register_player(name="p1", algorithm=HonestPlayer())
config.register_player(name="p2", algorithm=RLPokerAgent())
game_result = start_poker(config, verbose=1)
