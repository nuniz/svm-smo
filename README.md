# SVM Classifier with SMO Algorithm

This repository contains a Python implementation of a Support Vector Machine (SVM) classifier using the Sequential Minimal Optimization (SMO) algorithm [1] by Platt, using NumPy.

[1]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/nuniz/svm-smo.git
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
## Usage

Instantiate the SVM classifier with desired parameters and train it using your dataset. Once trained, you can make predictions on new data points.

```python
from svm import SVM

# Create SVM classifier
svm_classifier = SVM(kernel="rbf", max_iterations=200, eps=1e-5, cost=1.0, gamma=1.0)

# Train the classifier with your data (X_train, y_train)
svm_classifier.fit(X_train, y_train)

# Make predictions on new data (X_test)
predictions = svm_classifier.predict(X_test)