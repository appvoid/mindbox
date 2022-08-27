# MindBox
A minimal library to make neural networks easier.

# Quick Usage ⏱
```
from mindbox.core import network
model = network()
model.dataset({'i':([1],[0]),'o':[0,1]})
model.train()
print(model.predict([1]))
```

# Definition
MindBox: is a simple Python library to make neural networks in seconds!
MindBox takes inspiration from minimal libraries like brain.js to make ML fun again. You can just go ahead and train a model in literally the time you take to walk to your kitchen...

### Big thanks to:
Omar Aflak from Towards Data Science, which made the tutorial which powers a good part of this work! https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65