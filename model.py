from autograd import Value

class UnidimensionalNeuron:
    def __init__(self, lr = 0.025):
        self.weight = Value(20.0)
        self.bias = Value(10.0)
        self.learning_rate = Value(lr)

    def forward(self, input_):
        if not isinstance(input_, Value):
            raise ValueError(f"A unidimensional neuron only supports a Value as an input")
        return self.weight * input_ + self.bias 
        # mx = Value(self.weight.value*input.value, [self.weight, input])
        # return Value(mx.value + self.bias.value, [mx, self.bias])
    
    def descend(self):
        self.weight.value = self.weight.value - self.learning_rate.value*self.weight.gradient
        self.bias.value = self.bias.value - self.learning_rate.value*self.bias.gradient
        self.weight.gradient = 0
        self.bias.gradient = 0
