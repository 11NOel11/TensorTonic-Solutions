import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    y=np.array(x)

    sigmoid=1/(1+np.exp(-y))
    return sigmoid