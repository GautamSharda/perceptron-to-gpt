import math
import random
import matplotlib.pyplot as plt
import numpy as np
import heapq
from graphviz import Digraph

# Update graph at each step
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

    def __sub__(self, other_value, op='-'): # IMO makes sense to to do this via add too
        new_value = self.value - other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])
    
    def __truediv__(self, other_value, op='/'): # IMO makes sense to to do this via mul too
        new_value = self.value / other_value.value
        self.op = op
        other_value.op = op
        other_value.right = True
        return Value(new_value, [self, other_value])

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

def make_compute_graph(root_node, filename="compute_graph"):
    """
    Constructs and displays a graph visualization of the computation graph
    leading to the root_node, inferring operations from child nodes.

    Requires the 'graphviz' Python package and Graphviz software installed.
    Assumes the Value class stores the operation participated in on the
    operands ('op' attribute) and references operands in 'children'.

    Args:
        root_node: The final Value object (root of the graph to visualize).
        filename: Base name for the output graph file (without extension).
    """
    nodes_obj, edges_obj = set(), set()
    visited_ids = set()

    def build_graph_obj(v):
        if id(v) in visited_ids:
            return
        visited_ids.add(id(v))
        nodes_obj.add(v)
        # v.children stores the nodes that were inputs to the op creating v
        if hasattr(v, 'children') and v.children:
            for child in v.children:
                if child: # Skip None children (like in abs)
                    # Store edge as (child_object, parent_object)
                    edges_obj.add((child, v))
                    build_graph_obj(child) # Recurse on children

    build_graph_obj(root_node) # Start traversal from the root

    # Create the graphviz Digraph
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'}) # LR = Left to Right
    op_nodes_created = set() # Track operation nodes to avoid duplicates

    # Add Value nodes to the graph
    for n in nodes_obj:
        uid = str(id(n))
        # Format value and gradient nicely, handle potential NaN
        val_str = f"{n.value:.4f}" if isinstance(n.value, (int, float)) and not math.isnan(n.value) else str(n.value)
        grad_str = f"{n.gradient:.4f}" if isinstance(n.gradient, (int, float)) and not math.isnan(n.gradient) else str(n.gradient)
        label_str = f"val {val_str} | grad {grad_str}"
        dot.node(name=uid, label=label_str, shape='record')

    # Add Operation nodes and Edges
    for child, parent in edges_obj:
        child_uid = str(id(child))
        parent_uid = str(id(parent))

        # Infer the operation from the child's 'op' attribute
        # This 'op' signifies the operation that produced the 'parent'
        op_label = getattr(child, 'op', None) # Get op from child

        if op_label:
            # Create a unique ID for the operation node associated with the parent
            op_uid = parent_uid + '_op_' + op_label # Include op in ID for clarity if needed

            # Create the operation node only once per parent-operation pair
            if op_uid not in op_nodes_created:
                dot.node(name=op_uid, label=op_label, shape='ellipse')
                # Edge from operation node to the resulting parent Value node
                dot.edge(op_uid, parent_uid)
                op_nodes_created.add(op_uid)

            # Edge from the child Value node to the operation node
            dot.edge(child_uid, op_uid)
        else:
            # If child has no 'op', maybe it's an initial value?
            # Draw a direct edge for structure, though less informative.
            # This case might not happen if all ops set child.op correctly.
            dot.edge(child_uid, parent_uid)
            # print(f"Warning: Child node {child_uid} has no 'op' for edge to parent {parent_uid}")


    try:
        # Render the graph to a file and attempt to open it
        dot.render(filename, view=True, cleanup=True)
        print(f"Graph rendered to {filename}.png and opened.")
    except Exception as e:
        # Catch potential errors (e.g., graphviz not found)
        print(f"Error rendering graph (is graphviz installed and in PATH?): {e}")
        print("\nGraphviz Source:\n----------------")
        print(dot.source)
        print("----------------\n")
    while True:
        pass

def train_unidimensional_neuron(neuron, dataset, epochs, batch_size, accumulate=1):
    dataset_size = 0
    with open(dataset, "r") as f:
        dataset = [(Value(float(line.strip().split(",")[0])), Value(float(line.strip().split(",")[1]))) for line in f.readlines()]
        dataset_size = len(dataset)
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
        epoch_loss = 0 # sum of individual losses
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
            # make_compute_graph(avg_batch_loss)
            avg_batch_loss.backward()
            batches_processed += 1
            if batches_processed % accumulate == 0:
                neuron.descend()
        
        avg_example_loss_per_epoch = epoch_loss / dataset_size # this is probably a better measure than loss / batch
        heapq.heappush(epoch_losses, (avg_example_loss_per_epoch, e))
        
        print(f"e{e} -- weight: {neuron.weight.value}, bias: {neuron.bias.value}")
        print(f"avg loss: {avg_example_loss_per_epoch}")
    
    final_weight = neuron.weight.value
    final_bias = neuron.bias.value
    final_avg_loss = epoch_losses[0][0]
    best_epoch = epoch_losses[0][1]
    
    return {
        'epoch_losses': epoch_losses,
        'final_weight': final_weight,
        'final_bias': final_bias,
        'final_avg_loss': final_avg_loss,
        'best_epoch': best_epoch,
        'epochs': epochs
    }

def plot_unidimensional_neuron_training(results_list, labels=None, colors=None):
    """
    Plot multiple training results on the same graphs.
    
    Args:
        results_list: List of dictionaries, each containing training results
                     Each dict should have: 'epoch_losses', 'final_weight', 'final_bias', 'final_loss'
        labels: List of labels for each result set
        colors: List of colors for each result set
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(results_list))]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    # Plot 1: Loss curves
    plt.figure(figsize=(12, 8))
    
    for i, result in enumerate(results_list):
        epoch_losses = result['epoch_losses']
        # Sort by epoch number if using a heap
        sorted_losses = sorted(epoch_losses, key=lambda x: x[1])
        epoch_numbers = [pair[1] for pair in sorted_losses]
        losses = [pair[0] for pair in sorted_losses]
        
        plt.plot(epoch_numbers, losses, '-o', color=colors[i], label=labels[i], alpha=0.7)
    
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.legend()
    
    # Add trend lines
    for i, result in enumerate(results_list):
        epoch_losses = result['epoch_losses']
        sorted_losses = sorted(epoch_losses, key=lambda x: x[1])
        epoch_numbers = [pair[1] for pair in sorted_losses]
        losses = [pair[0] for pair in sorted_losses]
        
        if len(epoch_numbers) > 1:
            z = np.polyfit(epoch_numbers, losses, 1)
            p = np.poly1d(z)
            plt.plot(epoch_numbers, p(epoch_numbers), "--", color=colors[i], alpha=0.5,
                    label=f"{labels[i]} Trend: {z[0]:.6f}x + {z[1]:.6f}")
    
    plt.legend()
    
    # Plot 2: Learned functions
    plt.figure(figsize=(10, 6))
    x_values = np.linspace(0, 10, 100)
    plt.plot(x_values, x_values, 'k--', label='True function: f(x) = x')
    
    for i, result in enumerate(results_list):
        weight = result['final_weight']
        bias = result['final_bias']
        y_values = [weight * x + bias for x in x_values]
        plt.plot(x_values, y_values, color=colors[i], 
                label=f"{labels[i]}: y = {weight:.4f}x + {bias:.4f}")
    
    plt.title('Learned Functions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Model':<15} {'Final Weight':<15} {'Final Bias':<15} {'Final Loss':<15} {'Best Loss':<15}")
    print("-" * 80)
    
    for i, result in enumerate(results_list):
        best_loss = min(result['epoch_losses'], key=lambda x: x[0])[0]
        best_epoch = min(result['epoch_losses'], key=lambda x: x[0])[1]
        print(f"{labels[i]:<15} {result['final_weight']:<15.4f} {result['final_bias']:<15.4f} "
              f"{result['final_avg_loss']:<15.4f} {best_loss:<15.4f} (e{best_epoch})")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    EPOCHS = 72
    DATASET = "data.csv"
    labels = []

    sgd = UnidimensionalNeuron()
    result = train_unidimensional_neuron(neuron=sgd, dataset=DATASET, epochs=EPOCHS, batch_size=1)
    labels.append("sgd")

    bgd = UnidimensionalNeuron()
    result_two = train_unidimensional_neuron(bgd, DATASET, EPOCHS, batch_size=10)
    labels.append("bgd")

    mbgd = UnidimensionalNeuron()
    results_three = train_unidimensional_neuron(mbgd, DATASET, EPOCHS, batch_size=3)
    labels.append("mbgd")

    sgd_acc = UnidimensionalNeuron()
    results_four = train_unidimensional_neuron(sgd_acc, DATASET, EPOCHS, batch_size=1, accumulate=10)
    labels.append("sgd_acc")

    bgd_acc = UnidimensionalNeuron()
    results_five = train_unidimensional_neuron(bgd_acc, DATASET, EPOCHS, batch_size=10, accumulate=1)
    labels.append("bgd_acc")

    mbgd_acc = UnidimensionalNeuron()
    results_six = train_unidimensional_neuron(mbgd_acc, DATASET, EPOCHS, batch_size=3, accumulate=2)
    labels.append("mbgd_acc")

    plot_unidimensional_neuron_training([result, result_two, results_three, results_four, results_five, results_six], labels)
    