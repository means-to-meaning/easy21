import easy21
import utils
import feature_lookup as fl
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
import math
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


def get_feature_action(state, theta, feature_map):
    eps = 0.05
    action_type = np.random.choice(["explore", "exploit"], size=1, p=[eps, 1 - eps])[0]
    if action_type == "explore":
        action = np.random.choice([0, 1], size=1, p=[1 / 2, 1 / 2])[0]
    elif action_type == "exploit":
        q_0 = np.dot(feature_map[(0, state.dealers_card, state.players_sum)], theta)
        q_1 = np.dot(feature_map[(1, state.dealers_card, state.players_sum)], theta)
        action = np.argmax(np.array([q_0, q_1]))
    return int(action)

def run_episode(theta, lambd, feature_map):
    E_sa = np.zeros(36)
    gamma = 1
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = easy21.State(dealers_first_card, players_first_card)
    # initialize ACTION
    action = get_feature_action(state, theta, feature_map)
    assert action in [0, 1], str(action)
    game_history = []
    while not state.is_terminal:
        old_players_sum = state.players_sum
        old_dealers_card = state.dealers_card
        old_action = action
        next_state, reward = easy21.step(state, action)
        if not next_state.is_terminal:
            next_action = get_feature_action(state, theta, feature_map)
            delta = reward + \
                    gamma * \
                    np.dot(feature_map[next_action, next_state.dealers_card, next_state.players_sum], theta) - \
                    np.dot(feature_map[old_action, old_dealers_card, old_players_sum], theta)
        else:
            next_action = 0
            delta = reward + gamma * 0 - np.dot(feature_map[old_action, old_dealers_card, old_players_sum], theta)
        alpha = 0.01
        E_sa = gamma * lambd * E_sa + feature_map[(old_action, old_dealers_card, old_players_sum)]
        theta = theta + alpha * delta * E_sa
        state = next_state
        action = next_action
    return (theta, reward)

def convert_approx_qsa(feature_map, theta):
    Q_sa = np.zeros((2, 11, 22))
    for action in [0, 1]:
        for dealer in range(1, 11):
            for player in range(1, 22):
                Q_sa[(action, dealer, player)] = np.dot(feature_map[(action, dealer, player)], theta)
    return Q_sa

def main():
    n_episodes = 100000
    qsa_file_true = "results/Q_sa_true.pkl"
    qsa_file_sarsa_approx = "results/Q_sa_sarsa_approx.pkl"
    if os.path.exists(qsa_file_true):
        # evalutate existing policy
        truth_dict = pickle.load(open(qsa_file_true, "rb"))
        Q_sa_true = truth_dict["truth"]["Q_sa"]
        lambda_list = np.linspace(0, 1, num=6)
        lambda_detail_list = [0.0, 0.2, 1.0]
        # for llambda in lambda_list:
        lambd = 1
        mse_history = {}
        mse_final = {}
        feature_map = fl.get_feature()
        reward_hist = np.zeros(n_episodes)
        # sarsa_params = Params(N_0=100, llambda=llambda)
        for lambd in lambda_list:
            mse_ts = []
            theta = np.random.rand(36)
            logger.info("Starting SARSA game. lambda=" + str(lambd))
            wins_counter = 0
            for i in range(1, n_episodes + 1):
                theta, reward = run_episode(theta, lambd, feature_map)
                if reward == 1:
                    wins_counter += 1
                if i % 1000 == 0:
                    Q_sa = convert_approx_qsa(feature_map, theta)
                    mse = utils.mse_qsa(Q_sa_true, Q_sa)
                    logger.info("Playing game number: " + str(i) + ", MSE:" + str(mse))
                    print('Games {:d}, Wins {:0.2f}%'.format(i, (wins_counter / i) * 100))
                if lambd in lambda_detail_list:
                    reward_hist[i - 1] = reward
                    Q_sa = convert_approx_qsa(feature_map, theta)
                    mse = utils.mse_qsa(Q_sa_true, Q_sa)
                    mse_ts.append(mse)
            logger.info("Finished SARSA games.")
            if lambd in lambda_detail_list:
                Q_sa = convert_approx_qsa(feature_map, theta)
                mse = utils.mse_qsa(Q_sa_true, Q_sa)
                mse_history[lambd] = mse_ts
            mse_final[lambd] = mse
            if lambd == 0.2:
                pickle.dump({"sarsa_approx": {"Q_sa": Q_sa, "N_sa": [], "reward_hist": reward_hist,
                                        "mse_hist": mse_history[lambd]}}, open(qsa_file_sarsa_approx, "wb"))

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for llambda in [0.0, 1.0]:
            ax1.plot(mse_history[llambda], color=cm.hot(llambda-0.1), linestyle='-', label='lambda ' + str(llambda))
        x1, x2, y1, y2 = ax1.axis()
        ax1.axis((x1, x2, 0, 1))
        ax1.legend(loc='upper left')
        ax1.set_xlabel('# Episodes')
        ax1.set_ylabel('MSE')
        plt.show()

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(list(mse_final.keys()), list(mse_final.values()), 'ro')
        ax1.set_xlabel('lambda')
        ax1.set_ylabel('Final MSE - {} episodes'.format(n_episodes))
        plt.show()

if __name__ == "__main__":
    main()
