import easy21
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
import copy
np.random.seed(42)

console_log_legel = logging.INFO
file_log_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('easy21.log')
fh.setLevel(file_log_level)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(console_log_legel)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

class Params():

    def __init__(self, N_0=None, llambda=None):
        self.N_0 = N_0
        self.llambda = llambda
        self.gamma = 1
        dealers_card = range(1, 11)
        players_sum = range(-9, 32)
        states = [(a, b) for a in dealers_card for b in players_sum]
        self.Q_sa = {}
        self.N_sa = {}
        for state in states:
            self.N_sa[state] = {'hit': 0, 'stick': 0}
            self.Q_sa[state] = {'hit': 0, 'stick': 0}
        pol_keys = states
        pol_values = np.random.choice(['hit', 'stick'], size=len(states), p=[1 / 2, 1 / 2])
        self.optimal_policy = dict(zip(pol_keys, pol_values))
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
        eps = self.N_0 / (self.N_0 + self.N_s[state])
        action_type = np.random.choice(["explore", "exploit"], size=1, p=[eps, 1 - eps])[0]
        if action_type == "explore":
            action = np.random.choice(['hit', 'stick'], size=1, p=[1 / 2, 1 / 2])[0]
        elif action_type == "exploit":
            assert state in self.optimal_policy
            action = self.optimal_policy[state]
        return action

def get_action(state, Q_sa, N_sa, N_0=100):
    N_s = np.sum(N_sa[:, state.dealers_card, state.players_sum])
    eps = N_0 / (N_0 + N_s)
    action_type = np.random.choice(["explore", "exploit"], size=1, p=[eps, 1 - eps])[0]
    if action_type == "explore":
        action = np.random.choice([0, 1], size=1, p=[1 / 2, 1 / 2])[0]
    elif action_type == "exploit":
        action = np.amax(Q_sa[:, state.dealers_card, state.players_sum], axis=0)
    return action

def run_episode(Q_sa, N_sa, lambd):
    # dealers_card = range(1, 11)
    # players_sum = range(-9, 32)
    # states = [(a, b) for a in dealers_card for b in players_sum]
    E_sa = np.zeros((2, 11, 22))
    gamma = 1
    alpha = 1
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = easy21.State(dealers_first_card, players_first_card)
    # initialize ACTION
    action = get_action(state, Q_sa, N_sa)
    game_history = []
    while not state.is_terminal:
        old_players_sum = state.players_sum
        old_dealers_card = state.dealers_card
        old_action = action
        next_state, reward = easy21.step(state, action)
        if not next_state.is_terminal:
            next_action = get_action(state, Q_sa, N_sa)
            delta = reward + \
                    gamma * \
                    Q_sa[next_action, next_state.dealers_card, next_state.players_sum] - \
                    Q_sa[old_action, old_dealers_card, old_players_sum]
        else:
            next_action = 0
            delta = reward + gamma * 0 - Q_sa[old_action, old_dealers_card, old_players_sum]
        E_sa[old_action, old_dealers_card, old_players_sum] += 1
        Q_sa = Q_sa + alpha * delta * E_sa
        E_sa = gamma * lambd * E_sa
        state = next_state
        action = next_action
    return (Q_sa, N_sa)


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

def mse_policies(Q_sa1, Q_sa2):
    assert(Q_sa1.shape == Q_sa2.shape)
    res = np.sum(np.square(Q_sa1 - Q_sa2))
    res = res / Q_sa1.size
    return res

def main():
    policy_file = "data/Q_sa.pkl"
    if os.path.exists(policy_file):
        # evalutate existing policy
        [mc_Q_sa, mc_N_sa] = pickle.load(open(policy_file, "rb"))
        lambda_list = np.linspace(0, 1, num=6)
        lambda_detail_list = [0.0, 1.0]
        # for llambda in lambda_list:
        lambd = 1
        mse_history = {}
        mse_final = {}
        # sarsa_params = Params(N_0=100, llambda=llambda)
        n_iter = 10000
        for lambd in lambda_list:
            mse_ts = []
            logger.info("Starting SARSA game. lambda=" + str(lambd))
            Q_sa = np.zeros((2, 11, 22))
            N_sa = np.zeros((2, 11, 22), dtype=int)
            for i in range(1, n_iter + 1):
                Q_sa, N_sa = run_episode(Q_sa, N_sa, lambd)
                if i % 1000 == 0:
                    mse = mse_policies(mc_Q_sa, Q_sa)
                    logger.info("Playing game number: " + str(i) + ", MSE:" + str(mse))
                if lambd in lambda_detail_list:
                    mse = mse_policies(mc_Q_sa, Q_sa)
                    mse_ts.append(mse)
            logger.info("Finished SARSA games.")
            if lambd in lambda_detail_list:
                mse = mse_policies(mc_Q_sa, Q_sa)
                mse_history[lambd] = mse_ts
            mse_final[lambd] = mse
        plt.figure(1)
        for llambda in lambda_detail_list:
            plt.plot(mse_history[lambd], color=cm.hot(llambda-0.1), linestyle='-', label=str(llambda))
        plt.legend(loc='upper left')
        plt.show()

        plt.figure(2)
        plt.plot(list(mse_final.keys()), list(mse_final.values()), 'ro')
        plt.show()

if __name__ == "__main__":
    main()
