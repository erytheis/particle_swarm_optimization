import matplotlib
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# Global parameters
A = 0.7
B = 1.9
C = 2
Vmax = 10
num_of_particles = 20
num_of_iterations = 1000
x_lim = 10
y_lim = 10
benchmark_function = "rastrigin"  # rosenbrock/rastrigin

# Initialzing parameters
X = np.random.uniform(-2 * x_lim, 2 * x_lim, size = num_of_particles)
Y = np.random.uniform(-2 * y_lim, 2 * y_lim, size = num_of_particles)
Vx = np.random.rand(num_of_particles)
Vy = np.random.rand(num_of_particles)
Xpb = X  # particle best value's X
Ypb = Y  # particle best value's y
Fpb = np.zeros(num_of_particles)
F_hist = np.zeros((num_of_particles, num_of_iterations))
X_hist = np.zeros((num_of_particles, num_of_iterations))
Y_hist = np.zeros((num_of_particles, num_of_iterations))
Vx_hist = np.zeros((num_of_particles, num_of_iterations))
Vy_hist = np.zeros((num_of_particles, num_of_iterations))


def get_value(X, Y, benchmark_function):
    """
    Getting value from a benchmark function
    """
    if benchmark_function == "rastrigin":
        Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
        return Z

    elif benchmark_function == "rosenbrock":
        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
        return Z


"""
Main algorithm starts here
"""

# Calculate initial best values basing on random initial parameters
for idx in range(num_of_particles):
    Fpb[idx] = get_value(X[idx], Y[idx], benchmark_function)
Fgb = np.amin(Fpb)
Xgb = X[np.argmin(Fpb)]
Ygb = Y[np.argmin(Fpb)]

for epoch in range(num_of_iterations):
    if epoch % 5 == 0:
        print(
            "Epoch # " + str(epoch) + " best error = %.3f" % Fgb + " and corresponding X and Y (%.3f" %
            Xgb + ", %.3f" % Ygb + ")")

    for idx in range(num_of_particles):
        # update speeds
        Rgx = np.random.uniform()
        Rgy = np.random.uniform()
        Rpx = np.random.uniform()
        Rpy = np.random.uniform()
        Vx[idx] = A * Vx[idx] + B * Rpx * (Xpb[idx] - X[idx]) + C * Rgx * (Xgb - X[idx])
        Vy[idx] = A * Vy[idx] + B * Rpy * (Ypb[idx] - Y[idx]) + C * Rgy * (Ygb - Y[idx])

        # check if the velocities are larger then max
        if Vx[idx] > Vmax:
            Vx[idx] = Vmax
        if Vy[idx] > Vmax:
            Vy[idx] = Vmax

        # update positions
        X[idx] = X[idx] + Vx[idx]
        Y[idx] = Y[idx] + Vy[idx]

        # compute fitness
        F = get_value(X[idx], Y[idx], benchmark_function)

        # saving the values
        X_hist[idx][epoch] = X[idx]
        Y_hist[idx][epoch] = Y[idx]
        F_hist[idx][epoch] = F
        Vx_hist[idx][epoch] = Vx[idx]
        Vy_hist[idx][epoch] = Vy[idx]

        # check if new fitness is the best
        if F < Fpb[idx]:
            Fpb[idx] = F
            Xpb[idx] = X[idx]
            Ypb[idx] = Y[idx]

        if F < Fgb:
            Fgb = F_hist[idx][epoch]
            Xgb = X[idx]
            Ygb = Y[idx]


"""
Plotting methods
"""
def plot_results_animate_3d(benchmark_function, fig = plt.figure(), plot_function = True):
    """
    Plotting the animation of the results in 3D
    """
    ax = fig.gca(projection = '3d')

    if plot_function:
        if benchmark_function == "rastrigin":
            X = np.linspace(-x_lim, x_lim, 50)
            Y = np.linspace(-y_lim, y_lim, 50)
            X, Y = np.meshgrid(X, Y)
            ax.set_zlim(0, 100)
            Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
            ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.nipy_spectral, linewidth = 0.08,
                            antialiased = True)

        elif benchmark_function == "rosenbrock":
            X = np.linspace(-x_lim, x_lim, 50)
            Y = np.linspace(-y_lim, y_lim, 50)
            X, Y = np.meshgrid(X, Y)
            ax.set_zlim(0, 10000)
            Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
            ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.nipy_spectral, linewidth = 0.08,
                            antialiased = True)

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    sc = ax.scatter(X_hist, Y_hist, F_hist, c = "b")

    def animate(i):
        data = X_hist[:, i], Y_hist[:, i], F_hist[:, i]
        sc._offsets3d = data
        return sc,

    ani = matplotlib.animation.FuncAnimation(fig, animate,
                                             frames = num_of_iterations, interval = 100, blit = False)
    plt.show()

def plot_results_2d(benchmark_function, fig = plt.figure(), plot_function = True):
    """
    Plotting the static results in 2D with Heatmap
    """
    fig, ax = plt.subplots(1, 1)

    if plot_function:
        if benchmark_function == "rastrigin":
            X = np.linspace(-x_lim, x_lim, 50)
            Y = np.linspace(-y_lim, y_lim, 50)
            X, Y = np.meshgrid(X, Y)
            Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-y_lim, y_lim)
            ax.pcolor(X, Y, Z)

        elif benchmark_function == "rosenbrock":
            X = np.linspace(-x_lim, x_lim, 50)
            Y = np.linspace(-y_lim, y_lim, 50)
            X, Y = np.meshgrid(X, Y)
            Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-y_lim, y_lim)
            ax.pcolor(X, Y, Z, cmap = cm.nipy_spectral)

    c = ax.imshow(Z, cmap = cm.nipy_spectral)

    fig.colorbar(c, ax = ax)
    ax.scatter(X_hist[:, num_of_iterations - 1], Y_hist[:, num_of_iterations - 1], c = "y")
    plt.show()

def plot_results_animate_2d(benchmark_function, fig = plt.figure(), plot_function = True):
    """
    Plotting the animation of the results in 2D with Heatmap
    """
    ax = plt.axes(xlim = (-x_lim, x_lim), ylim = (-y_lim, y_lim))

    if plot_function:
        if benchmark_function == "rastrigin":
            X = np.linspace(-x_lim, x_lim, 50)
            Y = np.linspace(-y_lim, y_lim, 50)
            X, Y = np.meshgrid(X, Y)
            Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-y_lim, y_lim)
            ax.pcolor(X, Y, Z)

        elif benchmark_function == "rosenbrock":
            X = np.linspace(-x_lim, x_lim, 50)
            Y = np.linspace(-y_lim, y_lim, 50)
            X, Y = np.meshgrid(X, Y)
            Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-y_lim, y_lim)
            ax.pcolor(X, Y, Z, cmap = cm.nipy_spectral)

    c = ax.imshow(Z, cmap = cm.nipy_spectral)

    fig.colorbar(c, ax = ax)
    sc = ax.scatter(X_hist[:, num_of_iterations - 1], Y_hist[:, num_of_iterations - 1], c = "y")

    def animate(i):
        data = [list(X_hist[:, i]), list(Y_hist[:, i])]
        data = list(map(list, zip(*data)))

        sc.set_offsets(data)
        return sc,

    ani = matplotlib.animation.FuncAnimation(fig, animate,
                                             frames = num_of_iterations, interval = 100, blit = False)
    plt.show()



plot_results_animate_3d(benchmark_function, plot_function = False)
plot_results_animate_2d(benchmark_function)
