import math
import random
# Could pass a running gradient or store parents in nodes -- running seems more efficient for large NNs.
# Still need unit tests

            # In the case of the following local derivatives
            # add: d(z = x + y)/dx
            # sub: d(z = x - y)/dx
            # root: d(z)/dz
            # the derivative is 1
            # sub: d(z = x - y)/dy -- this is -1 since y is the right value!
            # mul: d(z = x*y)/dx -- this is y 
            # mul: d(z = x*y)/dy -- this is x
            # div: d(z = x/y)/dx -- this is 1/y
            # div: d(z = x/y)/dy -- this is d(x*y^-1)/dy -- so that's d(c*y^-1)/d -- so that's (c*-y^-2) -- that's (-c/y^2) -- (-x/y^2)

# If it works, it works...
class Value:
    def __init__(self, value, op=None, children=None):
        self.value = value
        self.op = op
        self.children = children
        self.gradient = 0.0
        self.right = False
    
    def __add__(self, other_value, op='+'):
        new_value = self.value + other_value.value
        other_value.right = True
        return Value(new_value, op, [self, other_value])

    def __mul__(self, other_value, op='*'):
        new_value = self.value * other_value.value
        other_value.right = True
        return Value(new_value, op, [self, other_value])

    def __sub__(self, other_value):
        negative = Value(-other_value.value)
        return self.__add__(negative, '-')
    
    def __truediv__(self, other_value):
        reciprocal = Value(1.0 / other_value.value)
        return self.__mul__(reciprocal, '/')
    
    def backward(self, other_value=None, chain_rule_grad=1.0):
        if self.op == "+":
            self.gradient = 1.0*chain_rule_grad
        if self.op == "-":
            if self.right:
                self.gradient = -1.0*chain_rule_grad
            else:
                self.gradient = 1.0*chain_rule_grad
        if self.op == '*':
            self.gradient = other_value.value*chain_rule_grad
        elif self.op == '/':
            if self.right:
                self.gradient = ((-other_value.value)/(self.value**2))*chain_rule_grad
            else:
                self.gradient = (1 / other_value.value)*chain_rule_grad
        else: # op will be None for root node
            self.gradient = 1.0
        if not self.children:
            return
        child_1, child_2 = self.children
        child_1.backward(child_2, self.gradient)
        child_2.backward(child_1, self.gradient)

class UnidimensionalNeuron:
    def __init__(self, lr = 0.1):
        self.weight = Value(random.random())
        self.bias = Value(random.random())
        self.learning_rate = Value(lr)

    def forward(self, input):
        if not isinstance(input, Value):
            raise ValueError(f"A unidimensional neuron only supports a Value as an input")
        return Value(self.weight.value*input.value + self.bias.value)
    
    def descend():
        self.weight.value = self.weight.value - self.learning_rate.value*self.weight.gradient.value

if __name__ == "__main__":
    neuron = UnidimensionalNeuron()
    with open("data.csv", "r") as f:
        data = [(Value(float(line.strip().split(",")[0])), Value(float(line.strip().split(",")[1]))) for line in f.readlines()]
    print(data)
    def avg_loss():
        total_loss = Value(0.0)
        for pair in data:
            x, y = pair
            print(type(y))
            total_loss += Value(abs((neuron.forward(x) - y).value))
        return total_loss / Value(len(data))
    
    epochs = 100
    for epoch in range(epochs): 
        avg_loss().backward()
        neuron.descend()

    print(avg_loss())         