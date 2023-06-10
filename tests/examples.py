import numpy as np


class quadratic1:
    def __init__(self, need_hessian=False):
        self.need_hessian = need_hessian

    def f(self, x):
        # x1^2+x2^2
        return x[0] ** 2 + x[1] ** 2

    def g(self, x):
        return np.array([2 * x[0], 2 * x[1]])

    def h(self, x):
        return np.array([[2, 0], [0, 2]])


class quadratic2:
    def __init__(self, need_hessian=False):
        self.need_hessian = need_hessian

    def f(self, x):
        return x[0] ** 2 + 100 * (x[1] ** 2)

    def g(self, x):
        return np.array([2 * x[0], 200 * x[1]])

    def h(self, x):
        return np.array([[2, 0], [0, 200]])


class quadratic3:
    def __init__(self, need_hessian=False):
        self.need_hessian = need_hessian
        self.P = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
        self.D = np.array([[100, 0], [0, 1]])
        self.Q = self.P.T @ self.D @ self.P

    def f(self, x):
        return x.T @ self.Q @ x

    def g(self, x):
        return 2 * self.Q @ x

    def h(self, x):
        return 2 * self.Q


class rosenbrock:
    def __init__(self, need_hessian=False):
        self.need_hessian = need_hessian

    def f(self, x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def g(self, x):
        return np.array([-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)])

    def h(self, x):
        return np.array([[-400 * (x[1] - 3 * x[0] ** 2) + 2, -400 * x[0]], [-400 * x[0], 200]])


class linear:
    def __init__(self, need_hessian=False):
        self.need_hessian = need_hessian
        self.a = np.array([1, 2])

    def f(self, x):
        return self.a.T @ x

    def g(self, x):
        return self.a

    def h(self, x):
        return np.zeros((2, 2))


class boyd:
    def __init__(self, need_hessian=False):
        self.need_hessian = need_hessian

    def f(self, x):
        return np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)

    def g(self, x):
        return np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                         3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])

    def h(self, x):
        return np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                          3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                         [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                          9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])
