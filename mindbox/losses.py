import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

# mae
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))

def mae_prime(y_true, y_pred):
    return np.sign(y_pred-y_true)/y_true.size

# cross entropy
def cross_entropy(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def cross_entropy_prime(y_true, y_pred):
    return -(y_true/y_pred - (1-y_true)/(1-y_pred))/y_true.size

# binary cross entropy
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return -(y_true/y_pred - (1-y_true)/(1-y_pred))/y_true.size

# categorical cross entropy
def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true*np.log(y_pred))/y_true.size

def categorical_cross_entropy_prime(y_true, y_pred):
    return -(y_true/y_pred)/y_true.size

# hinge loss
def hinge_loss(y_true, y_pred):
    return np.max(0, 1-y_true*y_pred)

def hinge_loss_prime(y_true, y_pred):
    return -y_true*(y_true*y_pred<1)