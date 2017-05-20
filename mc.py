import easy21
import numpy as np
np.random.seed(42)

k = 100
policy = {}
N_sa = {}
N_s = {}
Q_sa = {}
V_s = {}
N_0 = 1002

max_iter = 50000
loop_counter = 0
while loop_counter < max_iter:
    if loop_counter % 10000 == 0:
        print('*')
    de_first_card = easy21.get_cards()
    pl_first_card = easy21.get_cards()
    reward = None
    player_sum = pl_first_card
    game_history = []
    # game_history.append(((de_first_card, pl_first_card), None, 'hit'))
    state = (de_first_card, pl_first_card)
    while reward is None:
        if state not  in N_s:
            N_s[state] = 0
        eps = N_0 / (N_0 + N_s[state])
        action_type = np.random.choice(["explore", "exploit"], size = 1, p=[eps, 1-eps])[0]
        if action_type == "explore":
            action = np.random.choice(["hit", "stick"], size = 1, p=[1/2, 1/2])[0]
        else:
            q_max = min(Q_sa.values())
            action = min(Q_sa, key=Q_sa.get)
            for k, v in Q_sa.items():
                q_state = k[0]
                q_action = k[1]
                if q_state == state:
                    if v > q_max:
                        action = q_action

        ((de_first_card, player_sum), reward) = easy21.step(de_first_card, player_sum, action)
        if reward is None:
            game_history.append(((de_first_card, player_sum), 0, action))
        else:
            game_history.append(((de_first_card, player_sum), reward, action))

    # Update the policy estimate
    # print(game_history)
    G = game_history[-1][1]
    for record in game_history:
        state_action = (record[0],record[2])
        state = record[0]
        action = record[2]
        if state_action not in N_sa:
            N_sa[state_action] = 0
            Q_sa[state_action] = 0
        if state not in V_s:
            N_s[state] = 0
            V_s[state] = 0
        N_sa[state_action] += 1
        N_s[state] += 1
        Q_sa[state_action] = Q_sa[state_action] + (1/N_sa[state_action]) * (G - Q_sa[state_action])
        V_s[state] = V_s[state] + (1/N_s[state]) * (G - V_s[state])
    loop_counter += 1

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = []
Y = []
Z = []

for k, v in V_s.items():
    X.append(k[0])
    Y.append(k[1])
    Z.append(v)

lx = len(set(X))
ly = len(set(Y))
lz = len(Z)

XX = np.tile(np.array(list(set(X))), (ly, 1))
YY = np.tile(np.array(list(set(Y))), (lx, 1)).transpose()
ZZ = np.tile(np.array([0.0] * len(set(X))), (ly, 1))
for row in range(XX.shape[0]):
    for col in range(XX.shape[1]):
        hyp_state = (XX[row, col], YY[row, col])
        if hyp_state is (4,21):
            pass
        if hyp_state in V_s:
            ZZ[row, col] = V_s[hyp_state]

ax.plot_wireframe(XX, YY, ZZ)
ax.set_xlabel('Dealers card')
ax.set_ylabel('Sum')
ax.set_zlabel('Value')
plt.show()


# max_eval_iter = 10000
# while counter < max_eval_iter:
#     de_first_card = easy21.get_cards()
#     pl_first_card = easy21.get_cards()
#     reward = None
#     player_sum = pl_first_card
#     game_history = []
#     # game_history.append(((de_first_card, pl_first_card), None, 'hit'))
#     state = (de_first_card, pl_first_card)
#     while reward is None:
#         if state not  in N_s:
#             N_s[state] = 0
#         eps = N_0 / (N_0 + N_s[state])
#         action_type = np.random.choice(["explore", "exploit"], size = 1, p=[eps, 1-eps])[0]
#         if action_type == "explore":
#             action = np.random.choice(["hit", "stick"], size = 1, p=[1/2, 1/2])[0]
#         else:
#             q_max = min(Q_sa.values())
#             action = min(Q_sa, key=Q_sa.get)
#             for k, v in Q_sa.iteritems():
#                 q_state = k[0]
#                 q_action = k[1]
#                 if q_state == state:
#                     if v > q_max:
#                         action = q_action
#
#         ((de_first_card, player_sum), reward) = easy21.step(de_first_card, player_sum, action)
#         if reward is None:
#             game_history.append(((de_first_card, player_sum), 0, action))
#         else:
#             game_history.append(((de_first_card, player_sum), reward, action))
#     counter += 1