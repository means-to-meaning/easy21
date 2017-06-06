from .. import mc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_create_meshgrid():
    X = [0,2,4]
    Y = range(0,3)
    Z = {}
    for i in range(len(X)):
        Z[(X[i], Y[i])] = X[i] * Y[i]
    XX, YY, ZZ = mc.create_meshgrid(Z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(XX, YY, ZZ)
    ax.set_xlabel('Dealers card')
    ax.set_ylabel('Sum')
    ax.set_zlabel('Value')
    plt.show()