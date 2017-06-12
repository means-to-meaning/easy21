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
    while (cards_sum < 17 and cards_sum > 0) and cards_sum < players_sum:
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


def step(state):
    dealers_card = state[0]
    players_sum = state[1]
    players_sum = player_play(players_sum)
    # the only non-terminal state
    return((dealers_card, players_sum))


def eval_reward(state, dealers_sum):
    players_sum = state[1 ]
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

# def play_game():
#     de_first_card = get_cards()
#     pl_first_card = get_cards()
#     reward = None
#     player_sum = pl_first_card
#     game_history = []
#     # game_history.append(((de_first_card, pl_first_card), None, 'hit'))
#     while reward is None:
#         action = np.random.choice(['hit', 'stick'], size=1, p=[1/2, 1/2])[0]
#         ((de_first_card, player_sum), reward) = step(de_first_card, player_sum, action)
#         if reward is None:
#             game_history.append(((de_first_card, player_sum), 0, action))
#         else:
#             game_history.append(((de_first_card, player_sum), reward, action))
#     return game_history