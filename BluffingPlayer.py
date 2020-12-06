from pypokerengine.players import BasePokerPlayer

class BluffingPlayer(BasePokerPlayer):

    def declare_action(self, valid_actions, hole_card, round_state):
        action_call = valid_actions[1]
        action_raise = valid_actions[2]

        if action_raise:
            return action_raise['action'], action_raise['amount']['min']
        else:
            return action_call['action']. action_call['amount']

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass