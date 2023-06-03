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
save [hit enter]
📥 Model saved as: model.weights
# Sometime after saving...
model = network().load('your model name here or leave empty to use the default')
model.console()
```

### Description 📄
MindBox: is a simple Python library to make neural networks in seconds!
MindBox takes inspiration from minimal libraries like brain.js to make ML fun again. You can just go ahead and train a model in literally the time you take to walk to your kitchen...

### Big thanks to 😉
Omar Aflak from Towards Data Science, which made the tutorial which powers a good part of this work! https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65

### Short-Term Goals 🎯
- Documentation website
- Text classification
- Emoji support for Text Classification
- Improved code quality and algorithms performance
- Complete tests code

### Long-Term Goals 🎯
- Image Classification
- Possible Numba integration for training and inference
- Library website
