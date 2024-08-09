import numpy as np

class Weight:
  def __init__(self, weight):
    self.weight = weight

class Neuron:
  def __init__(self, value, activator = 0, bias = 0, weights = [Weight(0), Weight(0)]):
    self.value = value
    self.bias = bias
    self.activator = activator
    self.weights = weights
  def calculate(self, inputs):
    pass



def sigmoid(x):
    return 1/(1 + np.exp(-x))





class Layer:
  def __init__(self, neurons):
    self.neurons = neurons

class Network:
  def __init__(self, inputs: Layer, outputs: Layer, layers: list): # layers = [Layer (Neurons (Neuron1), Weight(Weight1) ), Layer2 etc.]
    self.inputs = inputs
    self.outputs = outputs
    self.layers = layers

    self.layers.append(outputs)
    self.layers.reverse()
    self.layers.append(inputs)
    self.layers.reverse()

  def set_inputs(self, new_inputs):
    self.inputs = new_inputs
    self.layers[0] = new_inputs
  def display(self, mode="verbose"):
    text = ""
    for layer in self.layers:
      for neuron in layer.neurons:
        match mode:
          case "verbose":
            text+="*" + str(neuron.activator) + "*  "
            text += str([w.weight for w in neuron.weights])
          case _:
            text+="*  "
        text+="\n"
      text+="\n"
      #text+=str([weight.weight for weight in layer.weights])
      text+="\n"
    return text


  def generate_weights(self):
    print(self.layers)
    for layeri in range(len(self.layers)-1):
      
      for neuroni in range(len(self.layers[layeri].neurons)):
        if len(self.layers[layeri].neurons[neuroni].weights) < len(self.layers[layeri+1].neurons):
          for neuronj in range(len(self.layers[layeri+1].neurons) - len(self.layers[layeri].neurons[neuroni].weights)):
            self.layers[layeri].neurons[neuroni].weights.append(Weight(1.0))



      '''if len(self.layers[layeri].weights) < len(self.layers[layeri+1].neurons):
        for neuron in range(len(self.layers[layeri+1].neurons) - len(self.layers[layeri].weights)):
          self.layers[layeri].weights.append(Weight(1.0))'''

  def calculate_cost(self, expected: int):
    cost = 0
    for i in self.layers[-1].neurons:
      #print(expected, ":", i.value)
      match expected:
        case i.value:
          cost += (i.activator - 1)**2
          #print("ye", i.activator)
        case _:
          #print("no", i.activator)
          cost += (i.activator - 0)**2
    return cost



  def train(self, expected_output=0):
    
    for layeri in range(1, len(self.layers)):
      activators = []
      biases = []
      for neuron in self.layers[layeri-1].neurons:
        activators.append(neuron.activator)
      for neuron in self.layers[layeri].neurons:
        biases.append(neuron.bias)
      weights = []
      for neuroni in range(len(self.layers[layeri].neurons)):
        weights.append([])
        for neuronj in range(len(self.layers[layeri-1].neurons)):

          weights[neuroni].append(self.layers[layeri-1].neurons[neuronj].weights[neuroni].weight)
      #print("-"*80)
      #print(weights, "\n", activators)
      #print(biases)
      result = np.array(list(map(sigmoid, np.add(np.dot(weights, activators), biases))))
      print(result)
      for i in range(len(self.layers[layeri].neurons)):
        self.layers[layeri].neurons[i].activator = result[i]



      #print("O:",weights,"\n",biases)
      new_weights = np.add(weights, np.negative(np.gradient(weights)))[1]
      new_biases = np.add(biases, np.negative(np.gradient(biases)))
      #print("N:",new_weights,"\n",new_biases)
      for neuroni in range(len(self.layers[layeri].neurons)):
        self.layers[layeri].neurons[neuroni].bias = new_biases[neuroni]
     
      for neuroni in range(len(self.layers[layeri].neurons)):
        for neuronj in range(len(self.layers[layeri-1].neurons)):
          #pass
          self.layers[layeri-1].neurons[neuronj].weights[neuroni].weight  = new_weights[neuroni][neuronj]
          #weights[neuroni].append(self.layers[layeri-1].neurons[neuronj].weights[neuroni].weight)



    return self.calculate_cost(expected_output)


    #for i in range(len(inputs)):
    #  self.inputs[i].activator = inputs[i]
    
