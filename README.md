# MindBox
A minimal library to make neural networks easier.

# Installation
```
pip install mindbox
```

### Quick Usage ⏱
```
from mindbox.core import network
model = network()
model.dataset({'i':([1],[0]),'o':[0,1]})
model.train()
model.console()
```
Later, if you want to use your new created network, you can just type save on the console (after testing with the `model.console()` method)
```
[type "save"] [hit enter]
📥 Model saved as: model.weights


# Sometime after saving...

from mindbox.core import network
import numpy as np
model = network().load('your model name here or leave empty to use the default')

def predict(sequence):
    return np.average(model.predict([sequence])) 

print(predict([1])) # expect a number near to 0
```

### Advanced 🤖
```
from mindbox.core import network
model = network('selu', debug=False) # supported activations are [sigmoid, tanh, relu, selu, gelu, swish]
model.dataset({'i':([1],[0]),'o':[0,1]})
model.layer(1, 5)
model.layer(5, 3)
model.layer(3, 1)
model.train(epochs=100, learning_rate=0.002)
output = model.predict([1])[0]
print(output)
```

### Description 📄
MindBox: is a simple Python library to make neural networks in seconds!
MindBox takes inspiration from minimal libraries like brain.js to make ML fun again. You can just go ahead and train a model in literally the time you take to walk to your kitchen...

The library automatically handles number of inputs for quick prototyping on real data. You can customize the model as you please:<br>

```
> model.train(args)
    epochs        (50)   : number of times the model will see the full dataset
    learning_rate (.3)   : how fast the model will learn, a lower number ensures avoiding the well-known overfitting
    counter       (100)  : how many epochs should train before printing results
    threshold     (None) : stops training when the error (loss) rate is less than the threshold
```
```
> activations
    [big architectures]
    sigmoid # this one works no matter the size of the architecture
    tanh    # works for "medium-size" architectures, about 25 or even more nodes or two layers of 12 inputs
    
    [small architectures] 
    all the activations left only work with medium or small number of nodes
    relu, selu, gelu, swish...
```

### Big thanks to 😉
Omar Aflak from Towards Data Science, which made the tutorial which powers a good part of this work! https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

### Short-Term Goals 🎯
- Text classification example
- Emoji support for Text Classification
- More examples: language models, computer vision, etc...
- Improved code quality and algorithms performance

### Long-Term Goals 🎯
- Library website
- Documentation website
