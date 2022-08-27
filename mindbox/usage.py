from core import network
model = network('sigmoid')
model.dataset({'i':([1],[0]),'o':[0,1]})
model.train(100)
print(model.accuracy)