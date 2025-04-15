import matplotlib.pyplot as plt
import numpy as np

def plot_training(results_list, labels=None, colors=None):
    if labels is None:
        labels = [f"Optimizer {i+1}" for i in range(len(results_list))]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
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
    print(f"{'Optimizer':<15} {'Final Weight':<15} {'Final Bias':<15} {'Final Loss':<15} {'Best Loss':<15}")
    print("-" * 80)
    
    for i, result in enumerate(results_list):
        best_loss = min(result['epoch_losses'], key=lambda x: x[0])[0]
        best_epoch = min(result['epoch_losses'], key=lambda x: x[0])[1]
        print(f"{labels[i]:<15} {result['final_weight']:<15.4f} {result['final_bias']:<15.4f} "
              f"{result['final_avg_loss']:<15.4f} {best_loss:<15.4f} (e{best_epoch})")
    
    plt.tight_layout()
    plt.show()