import matplotlib.pyplot as plt
import numpy as np
import heapq
from autograd import Value, make_computation_graph

def gradient_descent(neuron, dataset, epochs, batch_size, accumulate=1, viz=False):
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
            if viz:
                make_computation_graph(avg_batch_loss)
            avg_batch_loss.backward()
            if viz:
                make_computation_graph(avg_batch_loss)
            batches_processed += 1
            if batches_processed % accumulate == 0:
                neuron.descend()
        
        avg_example_loss_per_epoch = epoch_loss / dataset_size # this is probably a better measure than loss / batch
        heapq.heappush(epoch_losses, (avg_example_loss_per_epoch, e))
    
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
    