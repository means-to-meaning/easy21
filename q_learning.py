import easy21
import utils
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

console_log_level = logging.INFO
file_log_level = logging.DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('easy21.log')
fh.setLevel(file_log_level)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(console_log_level)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)


def run_episode(Q_sa, N_sa):
    gamma = 1
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = easy21.State(dealers_first_card, players_first_card)
    while not state.is_terminal:
        # initialize ACTION
        action = easy21.get_egreedy_action(state, Q_sa, N_sa)
        # action = easy21.get_random_action() # we could also run this off policy!!
        old_state = copy.deepcopy(state)
        next_state, reward = easy21.step(state, action)
        if state.is_terminal:
            target = reward
        else:
            target = reward + gamma * np.max(
                Q_sa[:, state.dealers_card, state.players_sum])
        N_sa[action, old_state.dealers_card, old_state.players_sum] += 1
        # alpha = 1 / N_sa[action, old_state.dealers_card, old_state.players_sum]
        alpha = 0.05
        Q_sa[action, old_state.dealers_card, old_state.players_sum] += + alpha * (target - Q_sa[action, old_state.dealers_card, old_state.players_sum])
    return (Q_sa, N_sa, reward)


def main():
    n_episodes = 100000
    qsa_file_true = "results/Q_sa_true.pkl"
    qsa_file_qlearning = "results/Q_sa_qlearning.pkl"
    if os.path.exists(qsa_file_true):
        # evalutate existing policy
        truth_dict = pickle.load(open(qsa_file_true, "rb"))
        Q_sa_true = truth_dict["truth"]["Q_sa"]
        mse_hist = np.zeros(n_episodes)
        reward_hist = np.zeros(n_episodes)
        # sarsa_params = Params(N_0=100, llambda=llambda)
        Q_sa = np.zeros((2, 11, 22))
        N_sa = np.zeros((2, 11, 22), dtype=int)
        wins_counter = 0
        for i in range(1, n_episodes + 1):
            Q_sa, N_sa, reward = run_episode(Q_sa, N_sa)
            if reward == 1:
                wins_counter += 1
            mse = utils.mse_qsa(Q_sa_true, Q_sa)
            print(mse)
            mse_hist[i-1] = mse
            reward_hist[i-1] = reward
            if i % 1000 == 0:
                logger.info("Playing game number: " + str(i) + ", MSE:" + str(mse))
                print('Games {:d}, Wins {:0.2f}%'.format(i, (wins_counter / i) * 100))
        pickle.dump({"qlearning": {"Q_sa": Q_sa, "N_sa": N_sa, "reward_hist": reward_hist,
                               "mse_hist": mse_hist}}, open(qsa_file_qlearning, "wb"))


if __name__ == "__main__":
    main()
