import math
import random
import matplotlib.pyplot as plt
import numpy as np
import heapq

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
# abs: d(z = |y|)/dy -- this is d(z = -y)/dy for y < 0 and d(z = y)/dy otherwise, which is -1 and 1 respectively

# TODO:
# Unit Tests
# Typing
# Order matters in batch processing -- it's worth asking why small to large / preserving order in this case works
# Also worth switching to randomly sampled batches
# And noting that accumulation tends to work better
# Neuron: LR variation = Adam
# Training: SGD / BGD / Mini-Batch Gradient Descent
# Batch size = how many examples before going backward. how large computaiton graph.
# In batch: batch size = N --> 1 backward pass.
# In stochastic: batch size = 1 --> N backward passes.
# In mini-batch: N > batch size > 1.
# Different from accumulation: this is about freq. of UPDATE WEIGHTS / epoch not freq. of GO BACKWARDS / epoch (batch size).
# So, you want to reset gradients after an update.

# How many forwards before the next backward (batch size)? How many backwards (batches) before the next update (accumulation size)?

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
            self.gradient += 1.0*chain_rule_grad
        elif self.op == "-":
            if self.right:
                self.gradient += -1.0*chain_rule_grad
            else:
                self.gradient += 1.0*chain_rule_grad
        elif self.op == '*':
            self.gradient += other_value.value*chain_rule_grad
        elif self.op == '/':
            if self.right:
                self.gradient += ((-other_value.value)/(self.value**2))*chain_rule_grad
            else:
                self.gradient += (1 / other_value.value)*chain_rule_grad
        elif self.op == "|":
            self.gradient += -1.0*chain_rule_grad if self.value < 0 else 1.0*chain_rule_grad
        else: # op will be None for root
            self.gradient += 1.0
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
        self.weight.gradient = 0
        self.bias.gradient = 0

def train_unidimensional_neuron(neuron, dataset, epochs, batch_size, accumulate=1):
    with open(dataset, "r") as f:
        dataset = [(Value(float(line.strip().split(",")[0])), Value(float(line.strip().split(",")[1]))) for line in f.readlines()]
    batches = []
    while dataset:
        batch = []
        for i in range(batch_size):
            if dataset:
                example = dataset.pop()
                batch.append(example)
        batches.append(batch)
    
    epoch_losses = []
    for e in range(epochs):
        print(f"e{e} -- weight: {neuron.weight.value}, bias: {neuron.bias.value}")
        epoch_loss = 0
        batches_processed = 0
        for batch in batches:
            total_batch_loss = Value(0.0)
            for example in batch:
                x, y = example
                prediction = neuron.forward(x)
                loss = abs((prediction - y))
                total_batch_loss += loss
                epoch_loss += loss.value
            
            avg_batch_loss = total_batch_loss / Value(batch_size)
            avg_batch_loss.backward()
            batches_processed += 1
            if batches_processed % accumulate == 0:
                neuron.descend()
        
        # Calculate and store average loss for this epoch
        avg_loss = epoch_loss / len(batches)
        heapq.heappush(epoch_losses, (avg_loss, e))
        
        print(f"e{e} -- weight: {neuron.weight.value}, bias: {neuron.bias.value}")
        print(f"avg loss: {avg_loss}")
    
    epoch_numbers = [pair[1] for pair in epoch_losses]
    # Show average loss graph per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, [pair[0] for pair in epoch_losses], 'b-o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    
    # Add a trend line to better visualize the overall direction
    if len(epoch_losses) > 1:
        z = np.polyfit(epoch_numbers, [pair[0] for pair in epoch_losses], 1)
        p = np.poly1d(z)
        plt.plot(epoch_numbers, p(epoch_numbers), "r--", alpha=0.7, 
                 label=f"Trend: {z[0]:.6f}x + {z[1]:.6f}")
        plt.legend()
    
    # Show final model parameters
    print(f"Final model parameters: weight={neuron.weight.value:.4f}, bias={neuron.bias.value:.4f}")
    print(f"Final average loss: {avg_loss}, epoch={epochs}")
    print(f"Best average loss: {epoch_losses[0][0]}, epoch={epoch_losses[0][1]}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Impl SGD, BGD, MBGD, with accumulation and non-accumulation for each -- that's 6 NNs in total.
# Same dataset size, epochs, and learning rates.

if __name__ == "__main__":
    EPOCHS = 72
    DATASET = "data.csv"

    # SGD
    sgd = UnidimensionalNeuron()
    # train_unidimensional_neuron(neuron=sgd, dataset=DATASET, epochs=EPOCHS, batch_size=1)

    # BGD
    bgd = UnidimensionalNeuron()
    # train_unidimensional_neuron(bgd, DATASET, EPOCHS, batch_size=10)

    # MBGD
    mbgd = UnidimensionalNeuron()
    # train_unidimensional_neuron(mbgd, DATASET, EPOCHS, batch_size=3)

    # SGD Acc
    sgd_acc = UnidimensionalNeuron()
    # train_unidimensional_neuron(sgd_acc, DATASET, EPOCHS, batch_size=1, accumulate=10)

    # BGD Acc
    bgd_acc = UnidimensionalNeuron()
    # train_unidimensional_neuron(bgd_acc, DATASET, EPOCHS, batch_size=10, accumulate=1)

    # MBGD_ACC
    mbgd_acc = UnidimensionalNeuron()
    # train_unidimensional_neuron(mbgd_acc, DATASET, EPOCHS, batch_size=3, accumulate=2)

# multiple neurons
# if __name__ == '__main__':
#     # Create multiple neurons with different learning rates
#     learning_rates = [0.001, 0.01, 0.025, 0.05, 0.1]
#     neurons = [UnidimensionalNeuron(lr=lr) for lr in learning_rates]
    
#     with open("data.csv", "r") as f:
#         data = [(Value(float(line.strip().split(",")[0])), Value(float(line.strip().split(",")[1]))) for line in f.readlines()]
    
#     # Track average losses per epoch for each neuron
#     all_epoch_losses = [[] for _ in learning_rates]
    
#     epochs = 72
#     for e in range(epochs):
#         print(f"Epoch {e}")
        
#         # Train each neuron and track losses
#         for i, neuron in enumerate(neurons):
#             total_loss = 0
            
#             for j, pair in enumerate(data):
#                 x, y = pair
#                 prediction = neuron.forward(x)
#                 loss = abs(prediction - y)
#                 total_loss += loss.value
                
#                 loss.backward()
#                 neuron.descend()
            
#             # Calculate and store average loss for this epoch
#             avg_loss = total_loss / len(data)
#             heapq.heappush(all_epoch_losses[i], (avg_loss, e))
            
#             print(f"  LR={learning_rates[i]}: weight={neuron.weight.value:.4f}, bias={neuron.bias.value:.4f}, avg_loss={avg_loss:.4f}")

#     # Visualization
#     plt.figure(figsize=(12, 8))
    
#     # Plot loss curves for each learning rate
#     for i, lr in enumerate(learning_rates):
#         # Extract epoch numbers and losses in order
#         sorted_losses = sorted(all_epoch_losses[i], key=lambda x: x[1])  # Sort by epoch number
#         epoch_numbers = [pair[1] for pair in sorted_losses]
#         losses = [pair[0] for pair in sorted_losses]
        
#         plt.plot(epoch_numbers, losses, '-o', label=f'LR={lr}', alpha=0.7)
    
#     plt.title('Average Loss per Epoch for Different Learning Rates')
#     plt.xlabel('Epoch')
#     plt.ylabel('Average Loss')
#     plt.grid(True)
#     plt.legend()
    
#     # Print final results for each learning rate
#     print("\nFinal Results:")
#     for i, lr in enumerate(learning_rates):
#         best_loss = all_epoch_losses[i][0][0]
#         best_epoch = all_epoch_losses[i][0][1]
#         print(f"LR={lr}: Best loss={best_loss:.4f} at epoch {best_epoch}")
#         print(f"  Final parameters: weight={neurons[i].weight.value:.4f}, bias={neurons[i].bias.value:.4f}")
    
#     # Plot the learned functions
#     plt.figure(figsize=(10, 6))
#     x_values = np.linspace(0, 10, 100)
#     plt.plot(x_values, x_values, 'k--', label='True function: f(x) = x')

#     for i, lr in enumerate(learning_rates):
#         y_values = [neurons[i].weight.value * x + neurons[i].bias.value for x in x_values]
#         plt.plot(x_values, y_values, label=f'LR={lr}: y = {neurons[i].weight.value:.4f}x + {neurons[i].bias.value:.4f}')
    
#     plt.title('Learned Functions for Different Learning Rates')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.grid(True)
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()
    