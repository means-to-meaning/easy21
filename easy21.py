import numpy as np

def get_card():
    lowest_card_value = 1
    highest_card_value = 11
    return np.random.randint(lowest_card_value, highest_card_value, size=1)[0]

def red_or_black(n=1):
    return np.random.choice(['red', 'black'], size=n, p=[1/3, 2/3])[0]

def player_play(pl_cards_sum):
    new_card = get_card()
    card_color = red_or_black()
    if card_color == 'red':
        pl_cards_sum = pl_cards_sum - new_card
    elif card_color == 'black':
        pl_cards_sum = pl_cards_sum + new_card
    return pl_cards_sum


def dealer_play(state):
    de_first_card = state[0]
    players_sum = state[1]
    cards_sum = de_first_card
    #
    while (cards_sum < 17 and cards_sum > 0):
        new_card = get_card()
        card_color = red_or_black()
        if card_color == 'red':
            cards_sum = cards_sum - new_card
        elif card_color == 'black':
            cards_sum = cards_sum + new_card
    return cards_sum


def is_player_bust(state):
    players_sum = state[1]
    return(players_sum > 21 or players_sum < 1)


def step(state, action):
    dealers_card = state[0]
    players_sum = state[1]
    dealers_sum = None
    reward = None
    if action == "hit":
        players_sum = player_play(players_sum)
        state = (dealers_card, players_sum)
        if is_player_bust(state):
            reward = -1
        else:
            reward = 0
    elif action == "stick":
        dealers_sum = dealer_play(state)
        reward = eval_reward(state[1], dealers_sum)
    if reward is None:
        raise Exception("Reward should not be None!")
    return (state, reward, dealers_sum)


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