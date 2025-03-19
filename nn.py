import math
import random
import matplotlib.pyplot as plt  # Add this import at the top
import numpy as np
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
# Typing

# Should op be the op used to create it or op that it was part of... -- should be op it was part of...
# Which means it should be set in the function, not at time of creation.

# What things should be Values? Well you need to compute the gradients backwards from the loss, so it seems like everything involved in computing the loss.
# But then you're summing the loss for each (input, output) pair in the dataset... should descend happen on a per data point basis? it's possible...
# I actually forget which way you did gradient descent -- yeah many variants, but let's on a per data point basis for simplicity of loss computation

# Ok how is it going up after reaching 0? Om the weights are overshooting bc gradient too high

# The absolute value needs to be backwardsed, the gradients should be reset after each backward pass.
class Value:
    def __init__(self, value, children=None):
        self.value = value
        self.op = None
        self.children = children
        self.gradient = 0.0
        self.right = False
    
    def __add__(self, other_value, op='+'):
        new_value = self.value + other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])

    def __mul__(self, other_value, op='*'):
        new_value = self.value * other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])

    def __sub__(self, other_value):
        negative = Value(-other_value.value)
        return self.__add__(negative, '-')
    
    def __truediv__(self, other_value):
        reciprocal = Value(1.0 / other_value.value)
        return self.__mul__(reciprocal, '/')

    def __abs__(self):
        self.op = '|'
        return Value(self.value*-1 if self.value < 0 else self.value, [self, None])
    
    def backward(self, other_value=None, chain_rule_grad=1.0):
        if self.op == "+":
            self.gradient = 1.0*chain_rule_grad
        elif self.op == "-":
            if self.right:
                self.gradient = -1.0*chain_rule_grad
            else:
                self.gradient = 1.0*chain_rule_grad
        elif self.op == '*':
            self.gradient = other_value.value*chain_rule_grad
        elif self.op == '/':
            if self.right:
                self.gradient = ((-other_value.value)/(self.value**2))*chain_rule_grad
            else:
                self.gradient = (1 / other_value.value)*chain_rule_grad
        elif self.op == "|":
            self.gradient = -1.0*chain_rule_grad if self.value < 0 else 1.0*chain_rule_grad
        else: # op will be None for root node
            self.gradient = 1.0
        if not self.children:
            return
        child_1, child_2 = self.children
        child_1.backward(child_2, self.gradient)
        child_2.backward(child_1, self.gradient) if child_2 else None

class UnidimensionalNeuron:
    def __init__(self, lr = 0.025):
        self.weight = Value(20.0)
        self.bias = Value(10.0)
        self.learning_rate = Value(lr)

    def forward(self, input):
        if not isinstance(input, Value):
            raise ValueError(f"A unidimensional neuron only supports a Value as an input")
        mx = Value(self.weight.value*input.value, [self.weight, input])
        return Value(mx.value + self.bias.value, [mx, self.bias])
    
    def descend(self):
        self.weight.value = self.weight.value - self.learning_rate.value*self.weight.gradient
        self.bias.value = self.bias.value - self.learning_rate.value*self.bias.gradient

if __name__ == "__main__":
    neuron = UnidimensionalNeuron()
    with open("data.csv", "r") as f:
        data = [(Value(float(line.strip().split(",")[0])), Value(float(line.strip().split(",")[1]))) for line in f.readlines()]
    def get_avg_loss():
        total_loss = Value(0.0)
        for pair in data:
            x, y = pair
            total_loss += abs((neuron.forward(x) - y).value) # have to reverse the abs function
        return total_loss / Value(len(data))
    
    # Track average losses per epoch for visualization
    epoch_losses = []
    epoch_numbers = []
    
    epoch = 72
    for e in range(epoch):
        print(f"e{e} -- weight: {neuron.weight.value}, bias: {neuron.bias.value}")
        total_loss = 0
        
        for i, pair in enumerate(data):
            x, y = pair
            prediction = neuron.forward(x)
            loss = abs((prediction - y))
            total_loss += loss.value
            
            loss.backward()
            neuron.descend()
            print(f"e{e}.{i} -- weight: {neuron.weight.value}, bias: {neuron.bias.value}")
        
        # Calculate and store average loss for this epoch
        avg_loss = total_loss / len(data)
        epoch_losses.append(avg_loss)
        epoch_numbers.append(e)
        
        print(f"e{e} -- weight: {neuron.weight.value}, bias: {neuron.bias.value}")
        print(f"avg loss: {avg_loss}")
    
    # Show average loss graph per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, epoch_losses, 'b-o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    
    # Add a trend line to better visualize the overall direction
    if len(epoch_losses) > 1:
        z = np.polyfit(epoch_numbers, epoch_losses, 1)
        p = np.poly1d(z)
        plt.plot(epoch_numbers, p(epoch_numbers), "r--", alpha=0.7, 
                 label=f"Trend: {z[0]:.6f}x + {z[1]:.6f}")
        plt.legend()
    
    # Show final model parameters
    print(f"Final model parameters: weight={neuron.weight.value:.4f}, bias={neuron.bias.value:.4f}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    