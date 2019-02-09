import matplotlib
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def plot_rastrigin(fig = plt.figure()):
    """
    Benchmark function
    """
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)

    X, Y = np.meshgrid(X, Y)
    Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.nipy_spectral, linewidth = 0.08, antialiased = True)
    # plt.savefig('rastrigin_graph.png')
    plt.show()


def get_value(X, Y):
    Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
    return Z


"""
Initializing parameters
"""
num_of_particles = 20
X = np.random.rand(num_of_particles)
Y = np.random.rand(num_of_particles)
Vx = np.zeros(num_of_particles)
Vy = np.zeros(num_of_particles)
A = 0.5
B = 2
C = 2

"""
Updating coordinates and speeds at each iteration
"""
for idx in range(num_of_particles):
    S = get_value(X[idx], Y[idx])
    R = np.random()
    Vx[idx] = A * Vx[idx] + B * R * (Xpb - S) + C * R * (Xgb - S)
    Vy[idx] = A * Vy[idx] + B * R * (Ypb - S) + C * R * (Ygb - S)



X = np.add(X, Vx)
Y = np.add(Y, Vy)
