import numpy as np


def line_search(func, grad, xk, pk, c1=0.01, alpha_init=1.0):
    alpha = alpha_init
    while True:
        func_xk = func(xk)
        func_xk_alpha_pk = func(xk + alpha * pk)
        grad_xk_dot_pk = np.dot(grad(xk).T, pk)

        if func_xk_alpha_pk > func_xk + c1 * alpha * grad_xk_dot_pk:
            alpha *= 0.5
        else:
            return alpha


class Optimizer:
    def __init__(self, method='gradient_descent', max_iter=1000, obj_tol=1e-5, param_tol=1e-5):
        self.method = method
        self.max_iter = max_iter
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.path = []

    def optimize(self, obj, x0):
        self.path = []
        if self.method == 'gradient_descent':
            return self.gradient_descent(obj, x0)
        elif self.method == 'newton':
            return self.newton(obj, x0)
        elif self.method == 'bfgs':
            return self.bfgs(obj, x0)
        elif self.method == 'sr1':
            return self.sr1(obj, x0)
        else:
            raise ValueError('Unknown method: ' + self.method)

    def gradient_descent(self, obj, x0):
        x = x0
        f_prev = obj.f(x)
        self.path.append((x, f_prev))

        for i in range(self.max_iter):
            grad = obj.g(x)
            if np.linalg.norm(grad) < self.obj_tol:
                return x, f_prev, True, self.path
            pk = -grad
            alpha = line_search(obj.f, obj.g, x, pk)
            x_new = x + alpha * pk
            f_new = obj.f(x_new)
            if abs(f_new - f_prev) < self.obj_tol or np.linalg.norm(
                    x_new - x) < self.param_tol:
                self.path.append((x_new, f_new))
                return x_new, f_new, True, self.path
            x = x_new
            f_prev = f_new
            self.path.append((x, f_new))
            print(f"Iteration {i}, location {x}, objective value {f_new}")
        return x, f_prev, False, self.path

    def newton(self, obj, x0):
        x = x0
        f_prev = obj.f(x)
        self.path.append((x, f_prev))
        for i in range(self.max_iter):
            grad = obj.g(x)
            if np.linalg.norm(grad) < self.obj_tol:
                return x, f_prev, True, self.path
            pk = -np.linalg.pinv(obj.h(x)).dot(grad)
            alpha = line_search(obj.f, obj.g, x, pk)
            x_new = x + alpha * pk
            f_new = obj.f(x_new)
            if abs(f_new - f_prev) < self.obj_tol or np.linalg.norm(
                    x_new - x) < self.param_tol:
                self.path.append((x_new, f_new))
                return x_new, f_new, True, self.path
            x = x_new
            f_prev = f_new
            self.path.append((x, f_new))
            print(f"Iteration {i}, location {x}, objective value {f_new}")
        return x, f_prev, False, self.path

    def bfgs(self, obj, x0):
        x = x0
        H = np.eye(len(x0))
        f_prev = obj.f(x)
        self.path.append((x, f_prev))
        for i in range(self.max_iter):
            grad = obj.g(x)
            if np.linalg.norm(grad) < self.obj_tol:
                return x, f_prev, True, self.path
            pk = -H.dot(grad)
            alpha = line_search(obj.f, obj.g, x, pk)
            x_new = x + alpha * pk
            f_new = obj.f(x_new)
            if abs(f_new - f_prev) < self.obj_tol or np.linalg.norm(
                    x_new - x) < self.param_tol:
                self.path.append((x_new, f_new))
                return x_new, f_new, True, self.path
            s = x_new - x
            y = obj.g(x_new) - grad
            if np.all(y == 0):
                return x_new, f_new, False, self.path
            rho = 1.0 / y.dot(s)
            H = (np.eye(len(x0)) - rho * np.outer(s, y)).dot(H).dot(
                np.eye(len(x0)) - rho * np.outer(y, s)) + rho * np.outer(s, s)
            x = x_new
            f_prev = f_new
            self.path.append((x, f_new))
            print(f"Iteration {i}, location {x}, objective value {f_new}")
        return x, f_prev, False, self.path

    def sr1(self, obj, x0):
        x = x0
        H = np.eye(len(x0))
        f_prev = obj.f(x)
        self.path.append((x, f_prev))
        for i in range(self.max_iter):
            grad = obj.g(x)
            if np.linalg.norm(grad) < self.obj_tol:
                return x, f_prev, True, self.path
            pk = -H.dot(grad)
            alpha = line_search(obj.f, obj.g, x, pk)
            x_new = x + alpha * pk
            f_new = obj.f(x_new)
            if abs(f_new - f_prev) < self.obj_tol or np.linalg.norm(
                    x_new - x) < self.param_tol:
                self.path.append((x_new, f_new))
                return x_new, f_new, True, self.path
            s = x_new - x
            y = obj.g(x_new) - grad
            if np.all(y == 0):
                return x_new, f_new, False, self.path
            H += np.outer(s - H.dot(y), s - H.dot(y)) / (s - H.dot(y)).dot(y)
            x = x_new
            f_prev = f_new
            self.path.append((x, f_new))
            print(f"Iteration {i}, location {x}, objective value {f_new}")
        return x, f_prev, False, self.path
