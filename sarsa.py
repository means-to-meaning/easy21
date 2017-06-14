import easy21
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
from mc import MCParams
np.random.seed(42)

class Params():

    def __init__(self, N_0=None, llambda=None):
        self.N_0 = N_0
        self.llambda = llambda
        dealers_card = range(1, 11)
        players_sum = range(1, 22)
        states = [(a, b) for a in dealers_card for b in players_sum]
        self.Q_sa = {}
        for state in states:
            # ALGO_CHOICE: player should never stick if dealer has higher card
            if state[0] >= state[1]:
                self.Q_sa[state] = {'hit': 0, 'stick': -1}
            else:
                self.Q_sa[state] = {'hit': 0, 'stick': 0}
        self.optimal_policy = dict.fromkeys(states, None)
        self.N_s = dict.fromkeys(states, 0)

    def __getitem__(self, state):
        params = {}
        if state in self.Q_sa:
            params["Q_sa"] = self.Q_sa[state]
        else:
            params["Q_sa"] = None
        return params

    def update_policy(self, state, action, reward):
        self.Q_sa[state][action] = reward
        action_rewards_dict = self.Q_sa[state]
        maxreward_action = max(action_rewards_dict, key=lambda i: action_rewards_dict[i])
        self.optimal_policy[state] = maxreward_action

    def get_action(self, state):
        dealers_card = state[0]
        players_sum = state[1]
        # ALGO_CHOICE: never stick if less than dealers card else we automatically loose
        if players_sum <= dealers_card:
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


def play_game(sarsa_params):
    alpha = 1
    dealers_card = range(1, 11)
    players_sum = range(1, 22)
    states = [(a, b) for a in dealers_card for b in players_sum]
    E_sa = {}
    for state in states:
        E_sa[state] = {'hit': 0, 'stick': 0}
    dealers_first_card = easy21.get_card()
    players_first_card = easy21.get_card()
    state = (dealers_first_card, players_first_card)
    reward = None
    game_history = []
    action = str(sarsa_params.get_action(state))
    while not (easy21.is_player_bust(state)):
        new_state, reward, dealers_sum = easy21.step(state, action)
        if action == "stick":
            game_history.append((state, action, reward, dealers_sum, new_state))
            break
        sarsa_params.N_s[state] += 1
        new_action = str(sarsa_params.get_action(new_state))
        new_new_state, new_reward, new_dealers_sum = easy21.step(new_state, new_action)
        delta = new_reward + sarsa_params.llambda * \
                             sarsa_params.Q_sa[new_state][new_action] - \
                sarsa_params.Q_sa[state][action]
        E_sa[state][action] += 1
        for a_state in states:
            for an_action in sarsa_params.Q_sa[a_state]:
                updated_policy_reward = sarsa_params.Q_sa[a_state][an_action] + alpha * delta * E_sa[a_state][an_action]
                sarsa_params.update_policy(a_state, an_action, updated_policy_reward)
        game_history.append((state, action, reward, dealers_sum, new_state))
        if new_action == "stick":
            game_history.append((new_state, new_action, new_reward, new_dealers_sum, new_new_state))
            break
        state = new_state
        action = new_action
    return (game_history, sarsa_params)


def create_meshgrid(Z):
    X = [state[0] for state in Z.keys()]
    Y = [state[1] for state in Z.keys()]
    X_vals = set(X)
    Y_vals = set(Y)
    lx = len(X_vals)
    ly = len(Y_vals)
    XX = np.tile(np.array(list(X_vals)), (ly, 1))
    YY = np.tile(np.array(list(Y_vals)), (lx, 1)).transpose()
    ZZ = np.tile(np.array([0.0] * len(X_vals)), (ly, 1))
    for k,v in Z.items():
        ZZ[ list(Y_vals).index(k[1]), list(X_vals).index(k[0])] = v
    return XX, YY, ZZ,


def plot_value_function(V_s):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    XX, YY, ZZ = create_meshgrid(V_s)
    ax.plot_wireframe(XX, YY, ZZ)
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 22, 2))
    ax.set_xlabel('Dealers card')
    ax.set_ylabel('Sum')
    ax.set_zlabel('Value')
    plt.show()


def action_to_int(action):
    if action == "hit":
        return 1
    if action == "stick":
        return -1


def plot_policy_function(policy):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    numeric_policy = {k: action_to_int(v) for k, v in policy.items()}
    XX, YY, ZZ = create_meshgrid(numeric_policy)
    ax.plot_wireframe(XX, YY, ZZ)
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 22, 2))
    ax.set_zticks([-1, 1])
    ax.set_xlabel('Dealers card')
    ax.set_ylabel('Sum')
    ax.set_zlabel('Action')
    plt.show()


def main():
    policy_file = "data/Q_sa.pkl"
    if os.path.exists(policy_file):
        # evalutate existing policy
        mc_params = MCParams()
        mc_params = pickle.load(open(policy_file, "rb"))
        optimal_policy = mc_params.optimal_policy
        sarsa_params = Params(N_0=100, llambda=0)
        n_iter = 1000
        for i in range(n_iter):
            game_history, sarsa_params = play_game(sarsa_params)
        diff = 0
        for state in sarsa_params.Q_sa:
            for action in sarsa_params.Q_sa[state]:
                diff = diff + (sarsa_params.Q_sa[state][action] - mc_params.Q_sa[state][action]) ** 2
        print(diff)

if __name__ == "__main__":
    main()
