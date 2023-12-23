import numpy as np
from .kernel import Kernel
from .utils import one_hot
from typing import Optional


class SVM:
    def __init__(self, kernel: str = "rbf", max_iterations: int = 200, eps: float = 1e-5, cost: float = 1.0,
                 gamma: float = 1.0):
        """
        Support Vector Machine (SVM) classifier.
        Solved by Sequential minimal optimization (SMO) algorithm.

        Parameters:
        - kernel (str): Kernel function type. Default is "rbf".
        - max_iterations (int): Maximum number of iterations for training. Default is 200.
        - eps (float): Tolerance for convergence. Default is 1e-5.
        - cost (float): Regularization parameter. Default is 1.0.
        - gamma (float): Kernel coefficient. Default is 1.0.
        """
        self.max_iterations = max_iterations
        self.eps = eps
        self.gamma = gamma
        self.cost = cost
        self.check_params()

        self.kernel = kernel

        self.alpha = np.empty((0, 0))
        self.b_model = np.empty(0)
        self.x = np.empty((0, 0))
        self.y = np.empty(0)
        self.support_vectors_ = np.empty((0, 0))

    def check_params(self):
        """
        Check parameters validity.

        Raises:
        - AssertionError: If parameter does not meet the requirements.
        """
        assert self.max_iterations > 0, f'max_iterations must be greater than 0, but obtained {self.max_iterations}'
        assert self.eps > 0.0, f"eps must be greater than 0, but obtained {self.eps}"
        assert self.cost >= 0.0, f"cost greater than or equal to 0, but obtained {self.cost}"

    def check_inputs(self, x: np.ndarray, y: Optional[np.ndarray]):
        """
        Check input data validity.

        Parameters:
        - x (np.ndarray): Input data.
        - y (Optional[np.ndarray]): Labels (optional).

        Raises:
        - AssertionError: If input data does not meet the requirements.
        """
        assert x.ndim == 2, "x must have 2 dimensions, num_examples x num_features"
        if y is not None:
            assert y.ndim == 1, "y must have 1 dimension, num_examples"
            assert y.size == x.shape[0], "y must have the same size as x/dim0"

    def decision_function(self, kernel_val, alpha, y, b_model):
        """
        Compute the decision function using the Kernel Trick.

        Parameters:
        - kernel_val: Kernel function values.

        Returns:
        - Decision values based on the SVM model.
        """
        # Calculate the decision function using the Kernel Trick
        self.support_vectors_ = np.dot(np.multiply(alpha, y).T, kernel_val)

        # Use the constant model bias for each class
        return self.support_vectors_.T + b_model

    def predict(self, x: np.ndarray):
        """
        Make predictions using the trained SVM model.

        Parameters:
        - x (np.ndarray): Input data.

        Returns:
        - Predicted labels.
        """
        self.check_inputs(x, y=None)
        self.check_params()

        kernel = Kernel(self.kernel, self.gamma)
        y = self.decision_function(kernel(self.x, x), self.alpha, self.y, self.b_model)
        return y.argmax(axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train the SVM model.

        Parameters:
        - x (np.ndarray): Input data.
        - y (np.ndarray): Labels.

        Returns:
        - None.
        """
        self.check_inputs(x, y)
        self.check_params()

        kernel = Kernel(self.kernel, self.gamma)
        kernel_xx = kernel(x, x)

        num_examples = x.shape[0]
        num_classes = np.unique(y).shape[0]
        y = one_hot(y, num_classes)
        self.x = x
        self.y = y
        self.alpha = np.zeros((num_examples, num_classes))
        self.b_model = np.zeros(num_classes)

        itr = 0
        while itr < self.max_iterations:
            # Counter to track the number of changes in the alpha values
            change_num = 0

            # Iterate over each example in the training data
            for j in range(num_examples):
                # Compute the error for the current example j
                e_j = self.decision_function(kernel_xx[j], self.alpha, self.y, self.b_model) - y[j]

                # Find indices of alpha values that violate the KKT conditions for example j
                idx = np.arange(num_classes)[
                    (np.abs(y[j] * e_j) > self.eps) * ((self.alpha[j] < self.cost) + (self.alpha[j] > 0))]

                # If there are violations, proceed with the SMO update
                if idx.size != 0:
                    # Select a random example i different from j
                    i = np.random.choice(np.setdiff1d(range(num_examples), j))

                    # Compute the error for example i
                    e_i = np.zeros(num_classes)
                    e_i[idx] = self.decision_function(kernel_xx[i, :], self.alpha[:, idx], y[:, idx],
                                                      self.b_model[idx]) - y[i, idx]

                    # Store the current alpha values for examples i and j
                    old_alpha_j = self.alpha[j, :]
                    old_alpha_i = self.alpha[i, :]

                    # Check if examples i and j belong to the same class
                    eq_arr = y[i, idx] == y[j, idx]

                    # Compute upper and lower bounds for alpha values
                    h_bound = np.zeros(num_classes)
                    l_bound = np.zeros(num_classes)
                    l_bound[idx] = np.maximum(0, self.alpha[i, idx] + self.alpha[j, idx] * (
                            2 * eq_arr - 1) - self.cost * eq_arr)
                    h_bound[idx] = np.minimum(self.cost,
                                              self.cost * (1 - eq_arr) + self.alpha[i, idx] + self.alpha[j, idx] * (
                                                      2 * eq_arr - 1))

                    # Update alpha values for examples i and j
                    idx = idx[h_bound[idx] != l_bound[idx]]
                    if idx.size == 0:
                        continue

                    eta = -kernel_xx[i, i] - kernel_xx[j, j] + 2 * kernel_xx[j, i]
                    if eta == 0:
                        continue

                    self.alpha[i, idx] -= ((y[i, idx] * (e_j[idx] - e_i[idx])) / eta).astype(float)
                    self.alpha[i, idx] = np.clip(self.alpha[i, idx], l_bound[idx], h_bound[idx]).astype(float)

                    # Update alpha value for example j
                    idx = idx[np.abs(self.alpha[i, idx] - old_alpha_i[idx]) > self.eps]
                    if idx.size == 0:
                        continue

                    self.alpha[j, idx] += self.y[i, idx] * self.y[j, idx] * (old_alpha_i[idx] - self.alpha[i, idx])

                    # Compute the change in alpha for the support vectors
                    delta_a = y[i, idx] * (self.alpha[i, idx] - old_alpha_i[idx]) * kernel_xx[i, i] + \
                              y[j, idx] * (self.alpha[j, idx] - old_alpha_j[idx]) * kernel_xx[j, j]

                    # Update the bias terms for the support vectors
                    b_1, b_2 = self.b_model[idx] - e_j[idx] - delta_a, self.b_model[idx] - e_i[idx] - delta_a
                    idx_b_1 = (0 < self.alpha[j, idx]) * (self.alpha[j, idx] < self.cost)
                    if np.size(idx_b_1) != 0:
                        self.b_model[idx[np.where(idx_b_1)]] = b_1[[np.where(idx_b_1)]]
                    idx_b_2 = (0 < self.alpha[i, idx]) * (self.alpha[i, idx] < self.cost)
                    if np.size(idx_b_2) != 0:
                        self.b_model[idx[np.where(idx_b_2)]] = b_2[[np.where(idx_b_2)]]
                    idx_b_3 = np.ones(idx.size) - idx_b_1 - idx_b_2
                    if np.size(idx_b_3) != 0:
                        self.b_model[idx[np.where(idx_b_3)]] = ((b_1 + b_2) / 2)[[np.where(idx_b_3)]]

                    # Increment the change counter
                    change_num += 1

            # If no changes were made in this iteration, increment the iteration counter
            if change_num == 0:
                itr += 1
