import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

def plot_training_loss_log(file_path="training_loss_history.txt", save_path=None):
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
    
    plt.figure(figsize=(12, 8))
    
    plt.semilogy(epochs, train_losses, linewidth=2, color='blue', 
                 label='Training Loss', marker='o', markersize=4)
    plt.semilogy(epochs, val_losses, linewidth=2, color='red', 
                 label='Validation Loss', marker='s', markersize=4)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (Log Scale)', fontsize=14)
    plt.title('Training and Validation Loss - Logarithmic Scale', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.minorticks_on()
    
    plt.xlim(1, max(epochs))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    load_dotenv()
    loss_history = os.getenv('LOSS_HISTORY')
    plot_training_loss_log(loss_history, "loss_plot_log.png")