import numpy as np # activation functions and its respectives derivatives

def tanh(x): return np.tanh(x);
def tanh_prime(x): return 1-np.tanh(x)**2;

def sigmoid(x): return 1/(1+np.exp(-x));
def sigmoid_prime(x): return sigmoid(x)*(1-sigmoid(x));

def relu(x): return np.maximum(0,x);
def relu_prime(x): return (x > 0).astype(x.dtype);

def srelu(x, a=1): return np.where(x >= 0, x, a*x);
def srelu_prime(x, a=1): return np.where(x >= 0, 1, a);

def leaky_relu(x): return np.maximum(0.01*x,x);
def leaky_relu_prime(x):
    grad = np.zeros(x.shape);
    grad[x>=0] = 1;
    grad[x<0] = 0.01;
    return grad;

def gelu(x): return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)));
def gelu_prime(x):
    return 0.5*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))+0.5*x*(1-np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))**2*np.sqrt(2/np.pi)*(1+0.044715*3*x**2);

def softplus(x): return np.log(1+np.exp(x));
def softplus_prime(x): return sigmoid(x);

def elu(x, alpha=1): return np.where(x >= 0, x, alpha*(np.exp(x)-1));
def elu_prime(x, alpha=1): return np.where(x >= 0, 1, alpha*np.exp(x));

def selu(x, alpha=1.67326, scale=1.0507): return scale*np.where(x >= 0, x, alpha*(np.exp(x)-1));
def selu_prime(x, alpha=1.67326, scale=1.0507): return scale*np.where(x >= 0, 1, alpha*np.exp(x));

def swish(x, beta=1): return x*sigmoid(beta*x);
def swish_prime(x, beta=1): return sigmoid(beta*x) + beta*x*sigmoid_prime(beta*x);

def hard_sigmoid(x): return np.clip(x*0.2+0.5, 0, 1);
def hard_sigmoid_prime(x): return np.where(x >= -2.5 and x <= 2.5, 0.2, 0);

def hard_tanh(x): return np.clip(x, -1, 1);
def hard_tanh_prime(x): return np.where(x >= -1 and x <= 1, 1, 0);

def linear(x): return x;
def linear_prime(x): return 1;

def softsign(x): return x/(1+np.abs(x));
def softsign_prime(x): return 1/(1+np.abs(x))**2;

def softmin(x): return -softmax(-x);
def softmin_prime(x): return -softmax_prime(-x);

def softmax(x): return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True);
def softmax_prime(x): return softmax(x)*(1-softmax(x));

def rectified_tanh(x): return np.tanh(np.sqrt(2/np.pi)*x);
def rectified_tanh_prime(x): return 1-np.tanh(np.sqrt(2/np.pi)*x)**2;

def gaussian(x): return np.exp(-x**2);
def gaussian_prime(x): return -2*x*np.exp(-x**2);

def logistic(x): return 1/(1+np.exp(-x));
def logistic_prime(x): return logistic(x)*(1-logistic(x));

def exp(x): return np.exp(x);
def exp_prime(x): return np.exp(x);