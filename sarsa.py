import numpy as np

class SARSAParams():

    def __init__(self):
        dealers_card = range(1, 11)
        players_sum = range(1, 22)
        states = [(a, b) for a in dealers_card for b in players_sum]
        self.N_0 = 100
        self.V_s = dict.fromkeys(states, 0)
        self.N_s = dict.fromkeys(states, 0)
        self.Q_sa = {}
        self.N_sa = {}
        for state in states:
            if state[0] > state[1]:
                self.Q_sa[state] = {'hit': 0, 'stick': -1}
            else:
                self.Q_sa[state] = {'hit': 0, 'stick': 0}
            self.N_sa[state] = {'hit': 0, 'stick': 0}
        self.optimal_policy = dict.fromkeys(states, None)

    def __getitem__(self, state):
        params = {}
        if state in self.V_s:
            params["V_s"] = self.V_s[state]
        else:
            params["V_s"] = None
        if state in self.N_s:
            params["N_s"] = self.N_s[state]
        else:
            params["N_s"] = None
        if state in self.Q_sa:
            params["Q_sa"] = self.Q_sa[state]
        else:
            params["Q_sa"] = None
        if state in self.N_sa:
            params["N_sa"] = self.N_sa[state]
        else:
            params["N_sa"] = None
        return params

    def iterate_value(self, state, reward):
        G = reward
        self.N_s[state] += 1
        self.V_s[state] = self.V_s[state] + (1 / self.N_s[state]) * (G - self.V_s[state])

    def iterate_policy(self, state, action, reward):
        # if state == (2,1):
        #     print(str(state), action, str(reward))
        G = reward
        self.N_sa[state][action] += 1
        self.Q_sa[state][action] = self.Q_sa[state][action] + (1 / self.N_sa[state][action]) * (
        G - self.Q_sa[state][action])
        action_rewards_dict = self.Q_sa[state]
        maxreward_action = max(action_rewards_dict, key=lambda i: action_rewards_dict[i])
        self.optimal_policy[state] = maxreward_action

    def get_action(self, state):
        dealers_card = state[0]
        players_sum = state[1]
        # never stick if less than dealers card else we automatically loose
        if players_sum < dealers_card:
            return 'hit'
        eps = self.N_0 / (self.N_0 + self.N_s[state])
        action_type = np.random.choice(["explore", "exploit"], size=1, p=[eps, 1 - eps])[0]
        if action_type == "explore":
            action = np.random.choice(['hit', 'stick'], size=1, p=[1 / 2, 1 / 2])[0]
        elif action_type == "exploit":
            if self.optimal_policy[state] is not None:
                action = self.optimal_policy[state]
            else:
                action = np.random.choice(['hit', 'stick'], size=1, p=[1 / 2, 1 / 2])[0]
        return action

def play_game(mc_params):
    dealers_first_card = easy21.get_card()
    players_first_card = easy21.get_card()
    state = (dealers_first_card, players_first_card)
    reward = None
    game_history = []
    while not easy21.is_player_bust(state):
        action = str(mc_params.get_action(state))
        game_history.append((state, action))
        if action == "stick":
            break
        else:
            state = easy21.step(state)
    if easy21.is_player_bust(state):
        reward = -1
    else:
        dealers_sum = easy21.dealer_play(state)
        reward = easy21.eval_reward(state, dealers_sum)
    if reward is None:
        raise Exception("Reward should not be None!")
    return (game_history, reward)