



import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os
def plot_training_loss_log(file_path="training_loss_history.txt", save_path=None):

    # Read the data
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Skip header lines
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            
            # Parse data
            parts = line.strip().split('\t')
            if len(parts) == 3:
                epoch = int(parts[0])
                train_loss = float(parts[1])
                val_loss = float(parts[2])
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot on log scale
    plt.semilogy(epochs, train_losses, linewidth=2, color='blue', 
                 label='Training Loss', marker='o', markersize=4)
    plt.semilogy(epochs, val_losses, linewidth=2, color='red', 
                 label='Validation Loss', marker='s', markersize=4)
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (Log Scale)', fontsize=14)
    plt.title('Training and Validation Loss - Logarithmic Scale', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, which='both')  # Grid for both major and minor ticks
    
    # Add minor ticks for better log scale readability
    plt.minorticks_on()
    
    # Set axis limits for better visualization
    plt.xlim(1, max(epochs))
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show final loss values
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"Total Epochs: {len(epochs)}")
    
    plt.show()

# Alternative version with both linear and log plots side by side
def plot_training_loss_comparison(file_path="training_loss_history.txt", save_path=None):
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split('\t')
            if len(parts) == 3:
                epoch = int(parts[0])
                train_loss = float(parts[1])
                val_loss = float(parts[2])
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale plot
    ax1.plot(epochs, train_losses, linewidth=2, color='blue', 
             label='Training Loss', marker='o', markersize=3)
    ax1.plot(epochs, val_losses, linewidth=2, color='red', 
             label='Validation Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Linear Scale)', fontsize=12)
    ax1.set_title('Linear Scale', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Logarithmic scale plot
    ax2.semilogy(epochs, train_losses, linewidth=2, color='blue', 
                 label='Training Loss', marker='o', markersize=3)
    ax2.semilogy(epochs, val_losses, linewidth=2, color='red', 
                 label='Validation Loss', marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax2.set_title('Logarithmic Scale', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.minorticks_on()
    
    plt.suptitle('Training and Validation Loss Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

# Usage examples:
if __name__ == "__main__":
    load_dotenv()
    loss_history = os.getenv('LOSS_HISTORY')
    plot_training_loss_log(loss_history, "loss_plot_log.png")
    
    # Comparison plot (both linear and log)
    # plot_training_loss_comparison("training_loss_history.txt", "loss_comparison.png")