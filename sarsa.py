import easy21
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
import utils
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


def run_episode(Q_sa, N_sa, lambd):
    E_sa = np.zeros((2, 11, 22))
    gamma = 1
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = easy21.State(dealers_first_card, players_first_card)
    # initialize ACTION
    action = easy21.get_action(state, Q_sa, N_sa)
    assert action in [0, 1], str(action)
    game_history = []
    while not state.is_terminal:
        old_players_sum = state.players_sum
        old_dealers_card = state.dealers_card
        old_action = action
        next_state, reward = easy21.step(state, action)
        if not next_state.is_terminal:
            next_action = easy21.get_action(state, Q_sa, N_sa)
            delta = reward + \
                    gamma * \
                    Q_sa[next_action, next_state.dealers_card, next_state.players_sum] - \
                    Q_sa[old_action, old_dealers_card, old_players_sum]
        else:
            next_action = 0
            delta = reward + gamma * 0 - Q_sa[old_action, old_dealers_card, old_players_sum]
        N_sa[old_action, old_dealers_card, old_players_sum] += 1
        alpha = 1 / N_sa[old_action, old_dealers_card, old_players_sum]
        E_sa[old_action, old_dealers_card, old_players_sum] += 1
        Q_sa = Q_sa + alpha * delta * E_sa
        E_sa = gamma * lambd * E_sa
        state = next_state
        action = next_action
    return (Q_sa, N_sa, reward)


def main():
    n_episodes = 100000
    qsa_file_true = "results/Q_sa_true.pkl"
    qsa_file_sarsa = "results/Q_sa_SARSA.pkl"
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
        reward_hist = np.zeros(n_episodes)
        # sarsa_params = Params(N_0=100, llambda=llambda)
        for lambd in lambda_list:
            mse_ts = []
            logger.info("Starting SARSA game. lambda=" + str(lambd))
            Q_sa = np.zeros((2, 11, 22))
            N_sa = np.zeros((2, 11, 22), dtype=int)
            wins_counter = 0
            for i in range(1, n_episodes + 1):
                Q_sa, N_sa, reward = run_episode(Q_sa, N_sa, lambd)
                if reward == 1:
                    wins_counter += 1
                if i % 1000 == 0:
                    mse = utils.mse_qsa(Q_sa_true, Q_sa)
                    logger.info("Playing game number: " + str(i) + ", MSE:" + str(mse))
                    print('Games {:d}, Wins {:0.2f}%'.format(i, (wins_counter / i) * 100))
                if lambd in lambda_detail_list:
                    reward_hist[i-1] = reward
                    mse = utils.mse_qsa(Q_sa_true, Q_sa)
                    mse_ts.append(mse)
            logger.info("Finished SARSA games.")
            if lambd in lambda_detail_list:
                mse = utils.mse_qsa(Q_sa_true, Q_sa)
                mse_history[lambd] = mse_ts
            mse_final[lambd] = mse
            if lambd == 0.2:
                pickle.dump({"lambda": {"Q_sa": Q_sa, "N_sa": N_sa, "reward_hist": reward_hist,
                                    "mse_hist": mse_history[lambd]}}, open(qsa_file_sarsa, "wb"))
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
