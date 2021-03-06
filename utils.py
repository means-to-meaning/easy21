import numpy as np
import matplotlib.pyplot as plt


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

def plot_value_function_qsa(Q_sa):
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

def mse_qsa(Q_sa1, Q_sa2):
    assert(Q_sa1.shape == Q_sa2.shape)
    res = np.sum(np.square(Q_sa1 - Q_sa2))
    res = res / Q_sa1.size
    return res