import numpy as np

class State():
    def __init__(self, dealers_card, players_sum):
        self.dealers_card = dealers_card
        self.players_sum = players_sum
        self.is_terminal = False

def get_first_card():
    lowest_card_value = 1
    highest_card_value = 11
    return int(np.random.randint(lowest_card_value, highest_card_value, size=1)[0])


def get_new_card():
    lowest_card_value = 1
    highest_card_value = 11
    new_card = int(np.random.randint(lowest_card_value, highest_card_value, size=1)[0])
    card_color = np.random.choice(['red', 'black'], size=1, p=[1/3, 2/3])[0]
    if card_color == 'red':
        return - new_card
    elif card_color == 'black':
        return new_card

def dealer_play(state):
    dealers_sum = state.dealers_card
    while (dealers_sum > 0 and dealers_sum < 17):
        dealers_sum += get_new_card()
    return dealers_sum


def is_player_bust(state):
    return(state.players_sum > 21 or state.players_sum < 1)


def eval_reward(players_sum, dealers_sum):
    is_dealer_bust = dealers_sum > 21 or dealers_sum < 1
    if is_dealer_bust:
        reward = 1
        return reward
    else:
        if players_sum > dealers_sum:
            reward = 1
            return reward
        elif players_sum < dealers_sum:
            reward = -1
            return reward
        elif players_sum == dealers_sum:
            reward = 0
            return reward


# actions: 1 - hit, 0 - stick
def step(state, action):
    reward = None
    if action == 1:
        state.players_sum += get_new_card()
        if is_player_bust(state):
            reward = -1
            state.is_terminal = True
        else:
            reward = 0
    else:
        dealers_sum = dealer_play(state)
        reward = eval_reward(state.players_sum, dealers_sum)
        state.is_terminal = True
    if reward is None:
        raise Exception("Reward should not be None!")
    return state, reward


