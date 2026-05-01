import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')  # or 'Qt5Agg', depending on what’s available
import matplotlib.pyplot as plt

def dot_plot(A, B):
    """
    Plots a scatter plot comparing corresponding elements of two matrices A and B
    around the reference line y = x.

    Parameters:
        A (numpy.ndarray): First n×n matrix
        B (numpy.ndarray): Second n×n matrix
    """
    # Check that the matrices have the same shape
    if A.shape != B.shape:
        raise ValueError("Matrices A and B must have the same shape.")

    # Flatten both matrices
    x = A.flatten()
    y = B.flatten()

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='royalblue', alpha=0.7, label='A vs B elements')

    # Reference line y = x
    plt.plot([min(x.min(), y.min()), max(x.max(), y.max())],
             [min(x.min(), y.min()), max(x.max(), y.max())],
             'k--', label='y = x')

    # Labels, legend, and styling
    plt.xlabel('tanimoto_scores')
    plt.ylabel('ms2_standard_scores')
    plt.title('Scatter Plot of Corresponding Elements from Two Matrices')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()