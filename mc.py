
import easy21
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
import copy
np.random.seed(42)



class MCParams():

    def __init__(self):
        self.Q_sa = np.zeros(2, 11, 21)
        self.N_sa = np.zeros(2, 11, 21)

    def __getitem__(self, state):
        params = {}
        params["Q_sa"] = self.Q_sa[:, state.dealers_card, state.players_sum]
        params["N_sa"] = self.N_sa[:, state.dealers_card, state.players_sum]
        return params

    def plot_value_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        XX, YY, ZZ = utils.create_meshgrid(self.V_s)
        ax.plot_wireframe(XX, YY, ZZ)
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(1, 22, 2))
        ax.set_xlabel('Dealers card')
        ax.set_ylabel('Sum')
        ax.set_zlabel('Value')
        plt.show()

    def update_action_value(self, state, action, value):
        self.Q_sa[action, state.dealers_card, state.players_sum] += value

    def update_state_count(self, state, action, value):
        self.N_sa[action, state.dealers_card, state.players_sum] = value

    def plot_policy_function(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        numeric_policy = {k: MCParams.action_to_int(v) for k, v in self.optimal_policy.items()}
        XX, YY, ZZ = utils.create_meshgrid(numeric_policy)
        ax.plot_wireframe(XX, YY, ZZ)
        ax.set_xticks(range(1, 11))
        ax.set_yticks(range(1, 22, 2))
        ax.set_zticks([-1, 1])
        ax.set_xlabel('Dealers card')
        ax.set_ylabel('Sum')
        ax.set_zlabel('Action')
        plt.show()

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


# def mc_policy_evaluation(mc_params, max_iter=10):
#     loop_counter = 1
#     while loop_counter <= max_iter:
#         if loop_counter % 10000 == 0:
#             print('Wins {:0.2f}%'.format(float(mc_params.n_wins)/mc_params.n_games))
#         game_history = play_game(mc_params)
#         for record in game_history:
#             state = record[0]
#             reward = record[2]
#             mc_params.iterate_value(state, reward)
#         loop_counter += 1
#     return mc_params





def main():
    # policy_file = "data/Q_sa.pkl"
    # if not os.path.exists(policy_file):
        # iterate policy
    Q_sa, N_sa = mc_policy_iteration(n_games=10000)
    plot_value(Q_sa)
    # pickle.dump([Q_sa, N_sa], open(policy_file, "wb"))
    # else:
    #     # evalutate existing policy
    #     [Q_sa, N_sa] = pickle.load(open(policy_file, "rb"))


if __name__ == "__main__":
    main()
