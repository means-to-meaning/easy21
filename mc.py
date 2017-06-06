import easy21
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
np.random.seed(42)

class MC_Params():

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

        def __getitem__(self, key):
            param_list = [self.N_0[key],
                          self.V_s[key],
                          self.N_s[key],
                          self.Q_sa[key],
                          self.N_sa[key]
                         ]
            for param in param_list:
                print(str(param))

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

    # def get_optimal_policy(self):
    #     optimal_policy = {}
    #     for state, action_rewards_dict in self.Q_sa.items():
    #         action = max(action_rewards_dict, key=lambda i: action_rewards_dict[i])
    #         optimal_policy[state] = action
    #     self.optimal_policy = optimal_policy


def play_game(mc_params):
    de_first_card = easy21.get_cards()
    pl_first_card = easy21.get_cards()
    reward = None
    player_sum = pl_first_card
    game_history = []
    # game_history.append(((de_first_card, pl_first_card), None, 'hit'))
    state = (de_first_card, pl_first_card)
    while True:
        action = str(mc_params.get_action(state))
        game_history.append((state, action))
        if action == "hit":
            state = easy21.step(state)
            players_sum = state[1]
            is_player_bust = players_sum > 21 or players_sum < 1
            if is_player_bust:
                reward = -1
                break
        elif action == "stick":
            dealers_sum = easy21.dealer_play(state)
            reward = easy21.eval_reward(state, dealers_sum)
            break
    if reward is None:
        raise Exception("Reward should not be None!")
    return (game_history, reward)


def mc_policy_iteration(N_0=100, max_iter=10000):
    mc_params = MC_Params()
    loop_counter = 0
    while loop_counter < max_iter:
        if loop_counter % 10000 == 0:
            print('*')
        game_history, reward = play_game(mc_params)
        # Update the policy estimate
        # if reward > -1:
        #     print("Positive reward:" + str(reward))
        for record in game_history:
            state = record[0]
            action = record[1]
            mc_params.iterate_value(state, reward)
            mc_params.iterate_policy(state, action, reward)
        loop_counter += 1
    return mc_params


def mc_policy_evaluation(mc_params, max_iter=10000):
    loop_counter = 0
    while loop_counter < max_iter:
        if loop_counter % 10000 == 0:
            print('*')
        game_history, reward = play_game(mc_params)
        for record in game_history:
            state = record[0]
            mc_params.iterate_value(state, reward)
        loop_counter += 1
    return mc_params


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
    ax.set_xlabel('Dealers card')
    ax.set_ylabel('Sum')
    ax.set_zlabel('Action')
    plt.show()


def main():
    # X = [0, 2, 4]
    # Y = range(0, 3)
    # Z = {}
    # for i in range(len(X)):
    #     Z[(X[i], Y[i])] = X[i] * Y[i]
    # XX, YY, ZZ = create_meshgrid(Z)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(XX, YY, ZZ)
    # ax.set_xlabel('Dealers card')
    # ax.set_ylabel('Sum')
    # ax.set_zlabel('Value')
    # plt.show()

    # X = [0, 2, 4]
    # Y = range(0, 3)
    # Z = {}
    # for i in range(len(X)):
    #     for j in range(len(Y)):
    #         Z[(X[i], Y[j])] = X[i] * Y[j]
    # XX, YY, ZZ = create_meshgrid(Z)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(XX, YY, ZZ)
    # ax.set_xlabel('Dealers card')
    # ax.set_ylabel('Sum')
    # ax.set_zlabel('Value')
    # plt.show()

    policy_file = "data/Q_sa.pkl"
    if not os.path.exists(policy_file):
        # iterate policy
        mc_params = mc_policy_iteration(max_iter=100000)
        plot_value_function(mc_params.V_s)
        plot_policy_function(mc_params.optimal_policy)
        pickle.dump(mc_params, open(policy_file, "wb"))
    else:
        # evalutate existing policy
        mc_params = pickle.load(open(policy_file, "rb"))
        optimal_policy = mc_params.optimal_policy
        mc_params = MC_Params()
        mc_params.optimal_policy = optimal_policy
        mc_params = mc_policy_evaluation(mc_params, max_iter=50000)
        plot_value_function(mc_params.V_s)


if __name__ == "__main__":
    main()
