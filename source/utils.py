import numpy as np

def one_hot(y: np.ndarray, num_classes: int):
    """
    Convert integer labels to one-hot encoded vectors.

    Parameters:
    - y (np.ndarray): Input array of integer labels.
    - num_classes (int): Number of classes.

    Returns:
    - One-hot encoded array with shape (y.shape + [num_classes]).
    """
    # Create an identity matrix with shape (num_classes, num_classes)
    identity_matrix = np.eye(num_classes)

    # Use the identity matrix to map each integer label to its one-hot encoded vector
    one_hot_encoded = identity_matrix[np.array(y).reshape(-1).astype(int)]

    # Transform the one-hot encoded vectors to binary vectors (-1 or 1)
    one_hot_encoded = (one_hot_encoded.reshape(list(y.shape) + [num_classes]) * 2) - 1

    return one_hot_encoded
