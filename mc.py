
import easy21
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import os
np.random.seed(42)



class MCParams():

    def __init__(self):
        dealers_card = range(1, 11)
        players_sum = range(1, 22)
        states = [(a, b) for a in dealers_card for b in players_sum]
        self.N_0 = 100
        self.V_s = dict.fromkeys(states, 0)
        self.N_s = dict.fromkeys(states, 0)
        self.Q_sa = {}
        self.N_sa = {}
        self.n_games = 0
        self.n_wins = 0
        for state in states:
            # # ALGO_CHOICE: player should never stick if dealer has higher card
            # if state[0] >= state[1]:
            #     self.Q_sa[state] = {'hit': 0, 'stick': -1}
            # else:
            #     self.Q_sa[state] = {'hit': 0, 'stick': 0}
            self.Q_sa[state] = {'hit': 0, 'stick': 0}
            self.N_sa[state] = {'hit': 0, 'stick': 0}
        self.optimal_policy = dict.fromkeys(states, None)

    def __getitem__(self, state):
        params = {}
        if state in self.V_s:
            params["V_s"] = self.V_s[state]
        else:
            params["V_s"] = None
        if state in self.N_s:
            params["N_s"] = self.N_s[state]
        else:
            params["N_s"] = None
        if state in self.Q_sa:
            params["Q_sa"] = self.Q_sa[state]
        else:
            params["Q_sa"] = None
        if state in self.N_sa:
            params["N_sa"] = self.N_sa[state]
        else:
            params["N_sa"] = None
        return params

    # def iterate_value(self, state, reward):
    #     G = reward
    #     self.N_s[state] += 1
    #     self.V_s[state] = self.V_s[state] + (1 / self.N_s[state]) * (G - self.V_s[state])

    def iterate_policy(self, state, action, reward):
        # if state == (2,1):
        #     print(str(state), action, str(reward))

        G = reward
        self.N_sa[state][action] += 1
        alpha = 1 / self.N_sa[state][action]
        self.Q_sa[state][action] = self.Q_sa[state][action] + alpha * (
        G - self.Q_sa[state][action])
        action_rewards_dict = self.Q_sa[state]
        # continuously update optimal policy
        maxreward_action = max(action_rewards_dict, key=lambda i: action_rewards_dict[i])
        self.optimal_policy[state] = maxreward_action

    def get_action(self, state):
        dealers_card = state[0]
        players_sum = state[1]
        # ALGO_CHOICE: never stick if less than dealers card else we automatically loose
        # if players_sum <= dealers_card:
        #     return 'hit'
        eps = self.N_0 / (self.N_0 + self.N_s[state])
        action_type = np.random.choice(["explore", "exploit"], size=1, p=[eps, 1 - eps])[0]
        if action_type == "explore":
            action = np.random.choice(['hit', 'stick'], size=1, p=[1 / 2, 1 / 2])[0]
        elif action_type == "exploit":
            # action_returns = self.Q_sa[state]
            # action = max(action_returns, key=lambda k: action_returns[k])
            if self.optimal_policy[state] is not None:
                action = self.optimal_policy[state]
            else:
                action = np.random.choice(['hit', 'stick'], size=1, p=[1 / 2, 1 / 2])[0]
        return action

    def calculate_optimal_value(self):
        for state in self.Q_sa:
            action_rewards_dict = self.Q_sa[state]
            max_value = max(action_rewards_dict.values())
            self.V_s[state] = max_value



    # def get_optimal_policy(self):
    #     optimal_policy = dict.fromkeys(self.V_s.keys(), None)
    #     for state, action_rewards in self.Q_sa.items():
    #         best_action = max(action_rewards, key=lambda k: action_rewards[k])
    #         self.optimal_policy[state] = best_action

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

    @staticmethod
    def action_to_int(action):
        if action == "hit":
            return 1
        if action == "stick":
            return -1

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


def play_game(mc_params):
    dealers_first_card = easy21.get_first_card()
    players_first_card = easy21.get_first_card()
    state = (dealers_first_card, players_first_card)
    reward = None
    game_history = []
    action = ""
    while not (easy21.is_player_bust(state) or action == "stick"):
        action = str(mc_params.get_action(state))
        old_state = state
        state, reward, dealers_sum = easy21.step(state, action)
        game_history.append((old_state, action, reward, dealers_sum, state))
    return (game_history)


def mc_policy_iteration(max_iter=10000):
    mc_params = MCParams()
    loop_counter = 1
    while loop_counter <= max_iter:
        if loop_counter % 10000 == 0:
            print('Wins {:0.2f}%'.format((float(mc_params.n_wins) / mc_params.n_games)*100))
        game_history = play_game(mc_params)
        # Update the policy estimate
        # if reward > -1:
        #     print("Positive reward:" + str(reward))
        mc_params.n_games += 1
        final_reward = game_history[-1][2]
        if final_reward == 1:
            mc_params.n_wins += 1
        for record in game_history:
            state = record[0]
            action = record[1]
            # TODO: add discounted rewards
            # reward =
            mc_params.iterate_policy(state, action, final_reward)
        loop_counter += 1
    return mc_params


def mc_policy_evaluation(mc_params, max_iter=10):
    loop_counter = 1
    while loop_counter <= max_iter:
        if loop_counter % 10000 == 0:
            print('Wins {:0.2f}%'.format(float(mc_params.n_wins)/mc_params.n_games))
        game_history = play_game(mc_params)
        for record in game_history:
            state = record[0]
            reward = record[2]
            mc_params.iterate_value(state, reward)
        loop_counter += 1
    return mc_params





def main():
    policy_file = "data/Q_sa.pkl"
    if not os.path.exists(policy_file):
        # iterate policy
        mc_params = mc_policy_iteration(max_iter=100000)
        mc_params.calculate_optimal_value()
        print(mc_params[(10, 12)])
        mc_params.plot_value_function()
        mc_params.plot_policy_function()
        pickle.dump(mc_params, open(policy_file, "wb"))
    else:
        # evalutate existing policy
        mc_params = pickle.load(open(policy_file, "rb"))
        # print(mc_params[(10, 12)])
        # optimal_policy = mc_params.optimal_policy
        # mc_params = MCParams()
        # mc_params.optimal_policy = optimal_policy
        # mc_params = mc_policy_evaluation(mc_params, max_iter=50000)
        mc_params.plot_value_function()
        mc_params.plot_policy_function()


if __name__ == "__main__":
    main()
