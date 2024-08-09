import cv2
import numpy as np
import glob
images = glob.glob("training/*")
training = []
for img_path in images:
  img = cv2.imread(img_path, 0) / 255.0
  print(img)
  data = img_path.split(".")[0].split("/")[1].split("_")
  training.append((img, int(data[1]), data[0])) # image, expected_result, test_pair_id


from package.package import *


hidden_layer = [Neuron(0, 0, weights = [Weight(float(i/5) - float(x*3)) for x in range(16)]) for i in range(16)]
hidden_layer2 = [Neuron(0, 0, weights = [Weight(float(i/5) - float(x*3)) for x in range(2)]) for i in range(16)]
#

inputLayer = Layer([Neuron(x,x, weights = [Weight(i/4) for i in range(-8,8)]) for i in training[0][0] for x in i])


outputLayer = Layer([Neuron(i, 0, weights = [Weight(0), Weight(0)]) for i in range(2)])

network = Network(inputLayer, outputLayer, [
  Layer(
    hidden_layer,
  ),
  Layer(
    hidden_layer2,
  ),
])
network.generate_weights()

print(network.display())

#print("cost: ", network.train(0))
for _ in range(25):
  for train in training:
    network.set_inputs(Layer([Neuron(x,x, weights = [Weight(x-8) for x in range(16)]) for i in train[0] for x in i]))
    network.generate_weights()
    #print(train[1])
    print("cost of pair ", train[2], ": ", network.train(train[1]))
'''
train = training[0]
for _ in range(10):
  network.set_inputs(Layer([Neuron(x,x, weights = [Weight(i/8) for i in range(-8,8)]) for i in train[0] for x in i]))
  network.generate_weights()
  #print(train[1])
  print("cost of pair ", train[2], ": ", network.train(train[1]))'''
