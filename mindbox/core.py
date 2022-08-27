# TODO: Implementet a way to gather all useful information into the model
''' 
    MindBox: is a simple Python library to make neural networks in seconds!

    Big thanks to:
    Omar Aflak and anyone who posts even something on StackOverflow to help me achieve this
'''
# modules
import os, numpy, mindbox.console as console; os.system('cls')
from mindbox.network import Network
from mindbox.fully_connected_layer import FCLayer
from mindbox.activation_layer import ActivationLayer
from mindbox.losses import mse, mse_prime

# main class
class network:
    def __init__(self, activation='tanh', debug=True):
        self.debug = debug # when enabled, prints useful messages to console
        self.net = Network(debug)
        self.activation = activation
        console.log('🧠 Neural Network loaded', self.debug)

    def encode(self, arr):
        # this method reduces the number of shapes to make numpy arrays more readable: no more [[[]]]
        result = []
        for i in range(0, len(arr)):
            result.append([arr[i]])
        return numpy.array(result)

    def parameters(self):
        # Formula for parameters calculation is: (consecutive dual sum + all parameters)
        # (input*hidden + hidden*hidden... + last hidden * output) + [inputs+hidden+output]
        # (h0∗h1 + h1∗h2) + [h0 + h1 + h2]
        # For example, on a simple setting, of 8 parameters:
        # params = (1*2 + 2*1) + [ 1 + 2 + 1 ] = (4) + [4] = 8
        output = self.net.layers[len(self.net.layers)-2].outputs_num

        allparams = [input.inputs_num for input in self.net.layers] # Iterates all layer's inputs
        allparams = list(filter(None, allparams)) # Removes activation layers from list
        allparams.append(output)

        dual_sum = []
        for i in range(0, len(allparams)):
            try:
                dual_sum.append(allparams[i]*allparams[i+1])
            except:
                break
        
        return (numpy.sum(dual_sum) + numpy.sum(allparams))

    def layer(self, inputs, outputs): 
        # TODO: Find a way to dinamically call these activations
        # for each layer we automatically add a new activation layer
        if (self.activation == 'sigmoid'): from mindbox.activations import sigmoid as activation, sigmoid_prime as prime
        elif (self.activation == 'tanh'): from mindbox.activations import tanh as activation, tanh_prime as prime
        elif (self.activation == 'relu'): from mindbox.activations import relu as activation, relu_prime as prime
        elif (self.activation == 'srelu'): from mindbox.activations import srelu as activation, srelu_prime as prime
        elif (self.activation == 'leaky_relu'): from mindbox.activations import leaky_relu as activation, leaky_relu_prime as prime
        elif (self.activation == 'gelu'): from mindbox.activations import gelu as activation, gelu_prime as prime
        elif (self.activation == 'softplus'): from mindbox.activations import softplus as activation, softplus_prime as prime
        elif (self.activation == 'elu'): from mindbox.activations import elu as activation, elu_prime as prime
        elif (self.activation == 'selu'): from mindbox.activations import selu as activation, selu_prime as prime
        elif (self.activation == 'swish'): from mindbox.activations import swish as activation, swish_prime as prime
        elif (self.activation == 'hard_sigmoid'): from mindbox.activations import hard_sigmoid as activation, hard_sigmoid_prime as prime
        elif (self.activation == 'hard_tanh'): from mindbox.activations import hard_tanh as activation, hard_tanh_prime as prime
        elif (self.activation == 'linear'): from mindbox.activations import linear as activation, linear_prime as prime
        elif (self.activation == 'softsign'): from mindbox.activations import softsign as activation, softsign_prime as prime
        elif (self.activation == 'softmin'): from mindbox.activations import softmin as activation, softmin_prime as prime
        elif (self.activation == 'softmax'): from mindbox.activations import softmax as activation, softmax_prime as prime
        elif (self.activation == 'rectified_tanh'): from mindbox.activations import rectified_tanh as activation, rectified_tanh_prime as prime
        elif (self.activation == 'gaussian'): from mindbox.activations import gaussian as activation, gaussian_prime as prime
        elif (self.activation == 'logistic'): from mindbox.activations import logistic as activation, logistic_prime as prime
        elif (self.activation == 'exp'): from mindbox.activations import exp as activation, exp_prime as prime
        self.net.add(FCLayer(inputs, outputs))
        self.net.add(ActivationLayer(activation, prime))
    
    def dataset(self, data):
        self.data = data
        self.number_of_inputs = len(data['i'][0])
        # TODO: Implement a tool for checking numbers intead of hardcoding an 's' letter
        s_for_input = ''
        if self.number_of_inputs > 1: s_for_input = 's'
        console.log('🔵 Dataset loaded with '+str(self.number_of_inputs)+' input'+s_for_input, self.debug)

    def train(self, epochs=50, learning_rate=.3, counter=100, threshold=None):
        console.log('🤖 Training using '+ self.activation + ' as activation', self.debug)

        if (not self.net.layers):
            console.log('🟣 Adding a layer since no layers added...', self.debug)
            self.layer(len(self.data['i'][0]),1) # this code tries to use a default layer when no provided
        self.net.use(mse, mse_prime)
        self.net.fit(self.encode(self.data['i']), self.encode(self.data['o']), int(epochs), learning_rate, counter, threshold)

        # sets accuracy percentage
        accuracy = round((0.1-self.net.error)*1000, 2)
        if accuracy < 0: accuracy = 0
        self.accuracy = accuracy

    def predict(self, input):
        return self.net.predict(self.encode(input)[0])

    # interface for inference testing
    def console(self,high_precission=False):
        try:
            console.log('🧮 Parameters: '+str(self.parameters()), self.debug)
            console.log('📈 Accuracy: '+str(self.accuracy)+'%', self.debug)
            
            if (self.number_of_inputs > 1 or self.number_of_inputs <= 0):
                console.log('📋 Type '+str(self.number_of_inputs)+' binary digits:', self.debug)
            else:
                console.log('📋 Type 1 binary digit:', self.debug)
            while 1:
                try:
                    data = input('>> ')
                    try:
                        if (data == 'exit'): 
                            os.system('cls'); break
                        elif (data == 'clear'): 
                            os.system('cls')
                        elif (data == 'save'): 
                            console.log('📥 Model saved as: model.weights')
                            self.save()
                        else:
                            data = [int(d) for d in data]
                            if (high_precission == True): # returns more digits: Eg. 0.00001231 instead of 0.1
                                print('<< '+str(self.predict(self.encode([data]))[0][0][0]))
                            else:
                                print('<< '+str(round(self.predict(self.encode([data]))[0][0][0],2)))
                    except:
                        console.log('Not a valid sequence!', self.debug)
                except:
                    console.log('🔵 Test finished', self.debug, False, False)
                    break
        except:
            console.log('🔴 Test cancelled!')
    
    # saves/loads weights as binary files
    def save(self, path='model.weights'):
        import pickle
        with open(path, "wb") as f:
            return pickle.dump(self, f)

    def load(self, path='model.weights'):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)