
import easy21
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
import copy
np.random.seed(42)


def run_episode(Q_sa, N_sa):
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = easy21.State(dealers_first_card, players_first_card)
    reward = None
    history = []
    action = ""
    while not state.is_terminal:
        action = easy21.get_action(state, Q_sa, N_sa)
        old_state = copy.deepcopy(state)
        state, reward = easy21.step(state, action)
        history.append((old_state, action, reward))
    return (history)


def iterate_policy(Q_sa, N_sa, episode_history):
    for record in episode_history:
        state = record[0]
        action = int(record[1])
        reward = record[2]
        as_idx = []
        # TODO: add discounted rewards
        # reward =
        G = episode_history[-1][2]
        N_sa[action, state.dealers_card, state.players_sum] += 1
        alpha = 1 / N_sa[action, state.dealers_card, state.players_sum]
        Q_sa[action, state.dealers_card, state.players_sum] += alpha * (G - Q_sa[action, state.dealers_card, state.players_sum])
    return Q_sa, N_sa


def mc_policy_iteration(n_games=1000, Q_sa_true=None):
    Q_sa = np.zeros((2, 11, 22))
    N_sa = np.zeros((2, 11, 22), dtype=int)
    wins_counter = 0
    games_counter = 0
    reward_hist = np.zeros(n_games)
    if Q_sa_true is not None:
        mse_hist = np.zeros(n_games)
    else:
        mse_hist = []
    while games_counter < n_games:
        if (games_counter+1) % 10000 == 0:
            print('Games {:d}, Wins {:0.2f}%'.format(games_counter+1,
                                                     (float(wins_counter) / (games_counter+1))*100))
        episode_history = run_episode(Q_sa, N_sa)
        reward = episode_history[-1][2]
        reward_hist[games_counter] = reward
        if reward == 1:
            wins_counter += 1
        Q_sa, N_sa = iterate_policy(Q_sa, N_sa, episode_history)
        if Q_sa_true is not None:
            mse = utils.mse_qsa(Q_sa, Q_sa_true)
            mse_hist[games_counter] = mse
        games_counter += 1
    print('Games {:d}, Wins {:0.2f}%'.format(games_counter,
                                             (float(wins_counter) / (games_counter)) * 100))
    return Q_sa, N_sa, reward_hist, mse_hist


def main():
    qsa_file_true = "results/Q_sa_true.pkl"
    qsa_file_mc = "results/Q_sa_MC.pkl"
    # let's get a good Q_sa estimate first using a large number of episodes!
    if not os.path.exists(qsa_file_true):
        Q_sa, N_sa, reward_hist, mse_hist = mc_policy_iteration(n_games=500000)
        utils.plot_value_function_qsa(Q_sa)
        utils.plot_policy_function(Q_sa)
        pickle.dump({"truth": {"Q_sa": Q_sa, "N_sa": N_sa, "reward_hist": reward_hist,
                               "mse_hist": mse_hist}}, open(qsa_file_true, "wb"))
    # let's also record a shorter run of MC for later examination of convergence
    else:
        truth_dict = pickle.load(open(qsa_file_true, "rb"))
        Q_sa_true = truth_dict["truth"]["Q_sa"]
        Q_sa, N_sa, reward_hist, mse_hist = mc_policy_iteration(n_games=100000, Q_sa_true=Q_sa_true)
        pickle.dump({"mc": {"Q_sa": Q_sa, "N_sa": N_sa, "reward_hist": reward_hist,
                               "mse_hist": mse_hist}}, open(qsa_file_mc, "wb"))


if __name__ == "__main__":
    main()
