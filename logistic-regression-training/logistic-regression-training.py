import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w=np.zeros(X.shape[1],dtype=float)
    b=0.0
    

    steps=0
    n=np.size(X)
    while steps<=1000:
        y_pred=_sigmoid(X@w+b)
        
        loss= (-1/n)*(np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))
        loss_grad_W=(1/n)*X.T@(y_pred-y)
        loss_grad_b=(1/n)*np.sum(y_pred-y)
        w=w-lr*loss_grad_W
        b=b-lr*loss_grad_b
        steps+=1
        

    return (w,b)
    