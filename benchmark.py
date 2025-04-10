import torch
import numpy as np
import heapq
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Imports for our implementation
from model import UnidimensionalNeuron as CustomNeuron # Alias to avoid name clash
from optimizer import gradient_descent
from utils.viz import plot_training

class UnidimensionalDataset(Dataset):
    def __init__(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                x, y = line.strip().split(',')
                data.append([float(x), float(y)])
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx, 0], self.data[idx, 1]

class UnidimensionalNeuron(torch.nn.Module):
    def __init__(self, lr=0.025):
        super().__init__()
        # Match the same initialization as original implementation
        self.linear = torch.nn.Linear(1, 1)
        self.linear.weight.data.fill_(20.0)
        self.linear.bias.data.fill_(10.0)
        self.learning_rate = lr
        
    def forward(self, x):
        return self.linear(x.unsqueeze(-1)).squeeze(-1)

def train_unidimensional_neuron(model, dataset_path, epochs, batch_size, accumulate=1):
    # Setup data
    dataset = UnidimensionalDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataset_size = len(dataset)
    
    # Training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate)
    epoch_losses = []
    
    for e in range(epochs):
        print(f"e{e} -- weight: {model.linear.weight.item()}, bias: {model.linear.bias.item()}")
        epoch_loss = 0
        batches_processed = 0
        optimizer.zero_grad()
        
        for x, y in dataloader:
            # Forward pass
            pred = model(x)
            loss = torch.abs(pred - y).mean()  # Average batch loss
            loss.backward()
            
            epoch_loss += loss.item() * len(x)  # Accumulate total loss
            batches_processed += 1
            
            if batches_processed % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle any remaining gradients
        if batches_processed % accumulate != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_example_loss_per_epoch = epoch_loss / dataset_size
        heapq.heappush(epoch_losses, (avg_example_loss_per_epoch, e))
        
        print(f"e{e} -- weight: {model.linear.weight.item()}, bias: {model.linear.bias.item()}")
        print(f"avg loss: {avg_example_loss_per_epoch}")
    
    return {
        'epoch_losses': epoch_losses,
        'final_weight': model.linear.weight.item(),
        'final_bias': model.linear.bias.item(),
        'final_avg_loss': epoch_losses[0][0],
        'best_epoch': epoch_losses[0][1],
        'epochs': epochs
    }

def plot_unidimensional_neuron_training(results_list, labels=None, colors=None):
    """
    Plot multiple training results on the same graphs.
    
    Args:
        results_list: List of dictionaries, each containing training results
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
    
    plt.title('Average Loss per Epoch (PyTorch Implementation)')
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
    
    plt.title('Learned Functions (PyTorch Implementation)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- PyTorch Implementation ---
    PYTORCH_EPOCHS = 72
    PYTORCH_DATASET = "data/data.csv" # Assuming data.csv is in data/
    pytorch_labels = []
    pytorch_results = []

    print("--- Running PyTorch Benchmark ---")

    # SGD
    sgd_torch = UnidimensionalNeuron()
    result_torch_sgd = train_unidimensional_neuron(sgd_torch, PYTORCH_DATASET, PYTORCH_EPOCHS, batch_size=1)
    pytorch_results.append(result_torch_sgd)
    pytorch_labels.append("sgd_torch")

    # BGD
    bgd_torch = UnidimensionalNeuron()
    result_torch_bgd = train_unidimensional_neuron(bgd_torch, PYTORCH_DATASET, PYTORCH_EPOCHS, batch_size=10)
    pytorch_results.append(result_torch_bgd)
    pytorch_labels.append("bgd_torch")

    # MBGD
    mbgd_torch = UnidimensionalNeuron()
    result_torch_mbgd = train_unidimensional_neuron(mbgd_torch, PYTORCH_DATASET, PYTORCH_EPOCHS, batch_size=3)
    pytorch_results.append(result_torch_mbgd)
    pytorch_labels.append("mbgd_torch")

    # SGD with accumulation
    sgd_acc_torch = UnidimensionalNeuron()
    result_torch_sgd_acc = train_unidimensional_neuron(sgd_acc_torch, PYTORCH_DATASET, PYTORCH_EPOCHS, batch_size=1, accumulate=10)
    pytorch_results.append(result_torch_sgd_acc)
    pytorch_labels.append("sgd_acc_torch")

    # BGD with accumulation
    bgd_acc_torch = UnidimensionalNeuron()
    result_torch_bgd_acc = train_unidimensional_neuron(bgd_acc_torch, PYTORCH_DATASET, PYTORCH_EPOCHS, batch_size=10, accumulate=1)
    pytorch_results.append(result_torch_bgd_acc)
    pytorch_labels.append("bgd_acc_torch")

    # MBGD with accumulation
    mbgd_acc_torch = UnidimensionalNeuron()
    result_torch_mbgd_acc = train_unidimensional_neuron(mbgd_acc_torch, PYTORCH_DATASET, PYTORCH_EPOCHS, batch_size=3, accumulate=2)
    pytorch_results.append(result_torch_mbgd_acc)
    pytorch_labels.append("mbgd_acc_torch")

    # Plot PyTorch results
    plot_unidimensional_neuron_training(pytorch_results, pytorch_labels)

    # --- Custom Autograd Implementation ---
    CUSTOM_EPOCHS = 72
    CUSTOM_DATASET = "data/data.csv"
    custom_labels = []
    custom_results = []

    print("\n--- Running Custom Autograd Implementation ---")

    # SGD
    sgd_custom = CustomNeuron()
    result_custom_sgd = gradient_descent(neuron=sgd_custom, dataset=CUSTOM_DATASET, epochs=CUSTOM_EPOCHS, batch_size=1)
    custom_results.append(result_custom_sgd)
    custom_labels.append("sgd_custom")

    # BGD
    bgd_custom = CustomNeuron()
    result_custom_bgd = gradient_descent(bgd_custom, CUSTOM_DATASET, CUSTOM_EPOCHS, batch_size=10)
    custom_results.append(result_custom_bgd)
    custom_labels.append("bgd_custom")

    # MBGD
    mbgd_custom = CustomNeuron()
    result_custom_mbgd = gradient_descent(mbgd_custom, CUSTOM_DATASET, CUSTOM_EPOCHS, batch_size=3)
    custom_results.append(result_custom_mbgd)
    custom_labels.append("mbgd_custom")

    # SGD with accumulation
    sgd_acc_custom = CustomNeuron()
    result_custom_sgd_acc = gradient_descent(sgd_acc_custom, CUSTOM_DATASET, CUSTOM_EPOCHS, batch_size=1, accumulate=10)
    custom_results.append(result_custom_sgd_acc)
    custom_labels.append("sgd_acc_custom")

    # BGD with accumulation
    bgd_acc_custom = CustomNeuron()
    result_custom_bgd_acc = gradient_descent(bgd_acc_custom, CUSTOM_DATASET, CUSTOM_EPOCHS, batch_size=10, accumulate=1)
    custom_results.append(result_custom_bgd_acc)
    custom_labels.append("bgd_acc_custom")

    # MBGD with accumulation
    mbgd_acc_custom = CustomNeuron()
    result_custom_mbgd_acc = gradient_descent(mbgd_acc_custom, CUSTOM_DATASET, CUSTOM_EPOCHS, batch_size=3, accumulate=2)
    custom_results.append(result_custom_mbgd_acc)
    custom_labels.append("mbgd_acc_custom")

    # Plot custom results (this function also prints the summary table)
    plot_training(custom_results, custom_labels)


    # --- Final Summary Tables ---

    print("\nPyTorch Implementation Results:")
    print("-" * 80)
    print(f"{'Model':<15} {'Final Weight':<15} {'Final Bias':<15} {'Final Loss':<15} {'Best Loss':<15}")
    print("-" * 80)

    for label, result in zip(pytorch_labels, pytorch_results):
        # Find best loss (lowest average loss) from the heap
        best_loss_info = min(result['epoch_losses'], key=lambda x: x[0])
        best_loss = best_loss_info[0]
        best_epoch = best_loss_info[1]
        # Final loss is the loss from the last epoch
        final_loss_info = sorted(result['epoch_losses'], key=lambda x: x[1])[-1]
        final_avg_loss = final_loss_info[0]

        print(f"{label:<15} {result['final_weight']:<15.4f} {result['final_bias']:<15.4f} "
              f"{final_avg_loss:<15.4f} {best_loss:<15.4f} (e{best_epoch})")
    print("-" * 80)

    # Print Custom Autograd summary table (same logic as plot_training)
    print("\Our Autograd Implementation Results:")
    print("-" * 80)
    print(f"{'Model':<15} {'Final Weight':<15} {'Final Bias':<15} {'Final Loss':<15} {'Best Loss':<15}")
    print("-" * 80)

    for label, result in zip(custom_labels, custom_results):
        best_loss_info = min(result['epoch_losses'], key=lambda x: x[0])
        best_loss = best_loss_info[0]
        best_epoch = best_loss_info[1]
        final_loss_info = sorted(result['epoch_losses'], key=lambda x: x[1])[-1]
        final_avg_loss = final_loss_info[0]

        print(f"{label:<15} {result['final_weight']:<15.4f} {result['final_bias']:<15.4f} "
              f"{final_avg_loss:<15.4f} {best_loss:<15.4f} (e{best_epoch})")
    print("-" * 80)