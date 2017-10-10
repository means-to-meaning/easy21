
import easy21
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
import copy
np.random.seed(42)


def plot_value(Q_sa):
    bestval = np.amax(Q_sa, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = range(1, 11)
    y = range(1, 22)
    X, Y = np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, np.rollaxis(bestval[1:, 1:], 1))
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 22, 2))
    ax.set_xlabel('Dealers first card')
    ax.set_ylabel('Players sum')
    ax.set_zlabel('Value')
    plt.show()

def plot_policy_function(Q_sa):
    bestaction = np.argmax(Q_sa, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = range(1, 11)
    y = range(1, 22)
    X, Y = np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, np.rollaxis(bestaction[1:, 1:], 1))
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 22, 2))
    ax.set_xlabel('Dealers first card')
    ax.set_ylabel('Players sum')
    ax.set_zlabel('Value')
    plt.show()

def get_action(state, Q_sa, N_sa, N_0=100):
    N_s = np.sum(N_sa[:, state.dealers_card, state.players_sum])
    eps = N_0 / (N_0 + N_s)
    action_type = np.random.choice(["explore", "exploit"], size=1, p=[eps, 1 - eps])[0]
    if action_type == "explore":
        action = np.random.choice([0, 1], size=1, p=[1 / 2, 1 / 2])[0]
    elif action_type == "exploit":
        action = np.amax(Q_sa[:, state.dealers_card, state.players_sum], axis=0)
    return action


def run_episode(Q_sa, N_sa):
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = easy21.State(dealers_first_card, players_first_card)
    reward = None
    history = []
    action = ""
    while not state.is_terminal:
        action = get_action(state, Q_sa, N_sa)
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


def mc_policy_iteration(n_games=1000):
    Q_sa = np.zeros((2, 11, 22))
    N_sa = np.zeros((2, 11, 22), dtype=int)
    wins_counter = 0
    games_counter = 0
    while games_counter < n_games:
        games_counter += 1
        if games_counter % 10000 == 0:
            print('Games {:d}, Wins {:0.2f}%'.format(games_counter,
                                                     (float(wins_counter) / games_counter)*100))
        episode_history = run_episode(Q_sa, N_sa)
        final_reward = episode_history[-1][2]
        if final_reward == 1:
            wins_counter += 1
        Q_sa, N_sa = iterate_policy(Q_sa, N_sa, episode_history)
    print('Games {:d}, Wins {:0.2f}%'.format(games_counter,
                                             (float(wins_counter) / games_counter) * 100))
    return Q_sa, N_sa


def main():
    policy_file = "data/Q_sa.pkl"
    if not os.path.exists(policy_file):
        # iterate policy
        Q_sa, N_sa = mc_policy_iteration(n_games=500000)
        plot_value(Q_sa)
        plot_policy_function(Q_sa)
        pickle.dump([Q_sa, N_sa], open(policy_file, "wb"))

    else:
        # evalutate existing policy
        [Q_sa, N_sa] = pickle.load(open(policy_file, "rb"))
        plot_value(Q_sa)
        plot_policy_function(Q_sa)



if __name__ == "__main__":
    main()
