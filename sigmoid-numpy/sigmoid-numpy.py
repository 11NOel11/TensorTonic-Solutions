import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    y=np.asarray(x, dtype=float)
    out=np.empty_like(y)
    pos = (y >= 0)
    neg = ~pos
    t = np.empty_like(y)
    np.exp(-y, out=t, where=pos)
    out[pos] = 1.0 / (1.0 + t[pos])
    

    np.exp(y, out=t, where=neg)
    out[neg] = t[neg] / (1.0 + t[neg])
    return out