import numpy as np

AVAILABLE_KERNELS = ['linear', 'rbf']

class Kernel:
    def __init__(self, ktype: str, gamma: float = 1.0):
        """
        Initialize a Kernel object.

        Parameters:
        - ktype (str): Type of kernel function ('linear' or 'rbf').
        - gamma (float): Kernel coefficient for 'rbf' kernel. Default is 1.0.
        """
        assert gamma > 0.0, f"gamma must be greater than 0, but obtained {gamma}"
        self.gamma = gamma

        if ktype == 'linear':
            self.kernel_function = self.linear
        elif ktype == 'rbf':
            self.kernel_function = self.rbf
        else:
            raise ValueError(f"kernel must be {' '.join(str(x) for x in AVAILABLE_KERNELS)}, but obtained {ktype}")
    
    def linear(self, x_a, x_b):
        """
        Linear kernel function.

        Parameters:
        - x_a (np.ndarray): Input data for the first set of points.
        - x_b (np.ndarray): Input data for the second set of points.

        Returns:
        - Linear kernel matrix.
        """
        return np.dot(x_a, x_b.T)

    def rbf(self, x_a, x_b):
        """
        Radial Basis Function (RBF) kernel function.

        Parameters:
        - x_a (np.ndarray): Input data for the first set of points.
        - x_b (np.ndarray): Input data for the second set of points.

        Returns:
        - RBF kernel matrix.
        """
        dist_matrix = np.linalg.norm(x_a[:, np.newaxis] - x_b, axis=2)
        return np.exp(-self.gamma * dist_matrix)

    def __call__(self, x_a, x_b):
        """
        Callable function to compute the kernel matrix.

        Parameters:
        - x_a (np.ndarray): Input data for the first set of points.
        - x_b (np.ndarray): Input data for the second set of points.

        Returns:
        - Computed kernel matrix using the selected kernel function.
        """
        return self.kernel_function(x_a, x_b)
