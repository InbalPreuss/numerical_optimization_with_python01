import numpy as np
import matplotlib.pyplot as plt


def plot_contour(obj, xlim, ylim, example_name, paths=None):
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)

    # Calculate the function value at each point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = obj.f(np.array([X[i, j], Y[i, j]]))

    # Plot
    plt.figure(figsize=(10, 8))
    if example_name == 'quadratic1' or example_name == 'quadratic2' or example_name == 'quadratic3' or example_name == 'linear':
        plt.contour(X, Y, Z, levels=50)
    elif example_name == 'boyd':
        plt.contour(X, Y, Z, levels=[0,2,5,10,20,50,100,200])
    else:
        plt.contour(X, Y, Z, levels=[20,30,40,50,80,100,200])

    plt.title('Contour plot of the objective function')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot the paths, if any
    if paths is not None:
        for path_name, path in paths.items():
            path = np.array(path)  # Convert list of tuples to a numpy array
            plt.plot(path[:, 0], path[:, 1], marker='o', label=path_name)
        plt.legend()

    plt.show()


def plot_function_values(function_values, labels):
    plt.figure(figsize=(10, 8))
    for i in range(len(labels)):
        plt.plot(function_values[i], label=labels[i], marker='o')
    plt.title('Function values at each iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.legend()
    plt.show()
