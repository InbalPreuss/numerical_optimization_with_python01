import unittest
import numpy as np
from examples import *
from src.utils import *
from src.unconstrained_min import *


class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        """Define all optimize methods method"""
        self.grad = Optimizer('gradient_descent', obj_tol=1e-12, param_tol=1e-8, max_iter=100)
        self.newton = Optimizer('newton', obj_tol=1e-12, param_tol=1e-8, max_iter=100)
        self.bfgs = Optimizer('bfgs', obj_tol=1e-12, param_tol=1e-8, max_iter=100)
        self.sr1 = Optimizer('sr1', obj_tol=1e-12, param_tol=1e-8, max_iter=100)
        self.functions = [("Newton", self.newton), ("Gradient Descent", self.grad), ("BFGS", self.bfgs),
                          ("sr1", self.sr1)]
        self.x0 = np.array([1.0, 1.0])

    def testing_method(self, objective_function, example_name):
        paths = {}  # Store paths of all methods
        function_values = []
        labels = []

        for (method_name, method) in self.functions:
            print("Testing method ", method_name)
            x_min, f_min, success, path = method.optimize(objective_function, self.x0)
            print(f'x_min: {x_min}, f_min: {f_min}, is_success: {success}, path: {path}')

            method_function_values = [p[1] for p in path]
            function_values.append(method_function_values)

            labels.append(method_name)

            # Storing the path for this method
            paths[method_name] = [p[0] for p in path]  # First element of each tuple in path is the location

        # After running all methods, plot the function values
        plot_function_values(function_values, labels)

        # Plot the contour with all paths
        if example_name == 'linear':
            plot_contour(objective_function, [-120, 10], [-220, 10], paths=paths, example_name=example_name)
        else:
            plot_contour(objective_function, [-10, 10], [-10, 10], paths=paths, example_name=example_name)


    def test_quadratic1(self):
        print("Testing quadratic1 function")
        self.x0 = np.array([1.0, 1.0])
        self.testing_method(quadratic1(need_hessian=True), 'quadratic1')

    def test_quadratic2(self):
        print("Testing quadratic2 function")
        self.x0 = np.array([1.0, 1.0])
        self.testing_method(quadratic2(need_hessian=True), 'quadratic2')

    def test_quadratic3(self):
        print("Testing quadratic3 function")
        self.x0 = np.array([1.0, 1.0])
        self.testing_method(quadratic3(need_hessian=True), 'quadratic3')

    def test_linear(self):
        print("Testing linear function")
        self.x0 = np.array([1.0, 1.0])
        self.testing_method(linear(need_hessian=True), 'linear')

    def test_boyd(self):
        print("Testing boyd function")
        self.x0 = np.array([1.0, 1.0])
        self.testing_method(boyd(need_hessian=True), 'boyd')

    def test_rosenbrock(self):
        print("Testing rosenbrock function")
        self.x0 = np.array([-1.0, 2.0])
        self.grad = Optimizer('gradient_descent', obj_tol=1e-12, param_tol=1e-8, max_iter=100000)
        self.functions = [("Gradient Descent", self.grad), ("Newton", self.newton), ("BFGS", self.bfgs),
                          ("sr1", self.sr1)]
        self.testing_method(rosenbrock(need_hessian=True), 'rosenbrock')


if __name__ == '__main__':
    unittest.main()
