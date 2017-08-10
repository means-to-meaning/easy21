import easy21
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
from mc import MCParams
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


def play_game(sarsa_params):
    dealers_card = range(1, 11)
    players_sum = range(-9, 32)
    states = [(a, b) for a in dealers_card for b in players_sum]
    E_sa = {}
    for state in states:
        E_sa[state] = {'hit': 0, 'stick': 0}
    # initialize STATE
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = (dealers_first_card, players_first_card)
    # initialize ACTION
    action = str(sarsa_params.get_action(state))
    reward = None
    game_history = []
    while not (easy21.is_player_bust(state)):
        new_state, reward, dealers_sum = easy21.step(state, action)
        sarsa_params.N_s[state] += 1
        sarsa_params.N_sa[state][action] += 1
        game_history.append((state, action, reward, dealers_sum, new_state))
        new_action = str(sarsa_params.get_action(new_state))
        assert new_state in sarsa_params.Q_sa, '{:s} not in'.format(str(new_state))
        delta = reward + \
                sarsa_params.gamma * \
                sarsa_params.Q_sa[new_state][new_action] - \
                sarsa_params.Q_sa[state][action]
        E_sa[state][action] += 1
        for a_state in states:
            assert a_state in sarsa_params.Q_sa, str(a_state)
            for an_action in sarsa_params.Q_sa[a_state]:
                assert an_action in sarsa_params.Q_sa[a_state]
                if sarsa_params.N_sa[a_state][an_action] > 0:
                    alpha = 1 / sarsa_params.N_sa[a_state][an_action]
                    updated_policy_reward = sarsa_params.Q_sa[a_state][an_action] + \
                                            alpha * delta * E_sa[a_state][an_action]
                    sarsa_params.update_policy(a_state, an_action, updated_policy_reward)
                    E_sa[a_state][an_action] = sarsa_params.gamma * sarsa_params.llambda * E_sa[a_state][an_action]
                elif sarsa_params.N_sa[a_state][an_action] == 0:
                    # just a logical placeholder to show that for never visited states-actions
                    #  we do nothing
                    pass
        if action == "stick" or reward == -1:
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

def mse_policies(policy1, policy2):
    mse = 0
    for state in policy1:
        for action in policy1[state]:
            mse = mse + (policy1[state][action] - policy2[state][action]) ** 2
    return(mse)

def main():
    policy_file = "data/Q_sa_.pkl"
    if os.path.exists(policy_file):
        # evalutate existing policy
        mc_params = pickle.load(open(policy_file, "rb"))
        lambda_list = np.linspace(0, 1, num=6)
        lambda_detail_list = [0.0, 1.0]
        # for llambda in lambda_list:
        llambda = 1
        mse_history = {}
        mse_final = {}
        sarsa_params = Params(N_0=100, llambda=llambda)
        n_iter = 100000
        for llambda in lambda_list:
            mse_ts = []
            logger.info("Starting SARSA game. lambda=" + str(llambda))
            for i in range(1, n_iter + 1):
                game_history, sarsa_params = play_game(sarsa_params)
                if i % 1000 == 0:
                    mse = mse_policies(mc_params.Q_sa, sarsa_params.Q_sa)
                    logger.info("Playing game number: " + str(i) + ", MSE:" + str(mse))
                if llambda in lambda_detail_list:
                    mse = mse_policies(mc_params.Q_sa, sarsa_params.Q_sa)
                    mse_ts.append(mse)
            logger.info("Finished SARSA games.")
            if llambda in lambda_detail_list:
                mse = mse_policies(mc_params.Q_sa, sarsa_params.Q_sa)
                mse_history[llambda] = mse_ts
            mse_final[llambda] = mse
        plt.figure(1)
        for llambda in lambda_detail_list:
            plt.plot(mse_history[llambda], color=cm.hot(llambda-0.1), linestyle='-', label=str(llambda))
        plt.legend(loc='upper left')
        plt.show()

        plt.figure(2)
        plt.plot(list(mse_final.keys()), list(mse_final.values()), 'ro')
        plt.show()

if __name__ == "__main__":
    main()
