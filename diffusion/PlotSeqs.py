import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio import SeqIO
from src import GetFasta, GetOneHot, GetNumeric
import torch
import os
from dotenv import load_dotenv


st_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU',
         'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
         'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
         'TYR', 'VAL', 'GAP']


plot = True
file_path = '../'
typ = 'gen_around_WT'

load_dotenv()
#dat_path = os.getenv('DATA_PATH')
#file_name_base = os.getenv('FILE_NAME')

# Check if environment variables exist
#if dat_path is None:
    #raise ValueError("DATA_PATH environment variable is required")
#if file_name_base is None:
    #raise ValueError("FILE_NAME environment variable is required")

#file_name = os.path.join(dat_path, file_name_base + '.fasta')
file_name = os.getenv('FILE1_PATH')

device = 'cpu'
max_seq_len = 146
seq_list, len_list = GetFasta(file_name)

one_hot_batch = GetOneHot(seq_list, max_seq_len, device='cpu')
numeric_batch = GetNumeric(seq_list, max_seq_len, device='cpu')

max_sequences = 1800
if numeric_batch.shape[0] > max_sequences:
    numeric_batch = numeric_batch[:max_sequences]
    one_hot_batch = one_hot_batch[:max_sequences]
    print(f"Limited to first {max_sequences} sequences out of {len(seq_list)} total")
else:
    print(f"Dataset has {numeric_batch.shape[0]} sequences (less than {max_sequences})")

n_samps = numeric_batch.shape[0]

if plot:
    cmap = "tab20c"
    lc = 'darkblue'
    ts = 13
    plot_fn = 'Sequences_aligned_' + typ + f'_first_{n_samps}.png'  # Updated filename
    
    fig = plt.figure(figsize=[8,3.5])

    plt.imshow(numeric_batch, aspect='auto', interpolation=None, origin='lower',
               cmap=cmap, alpha=0.8)

    cbar = plt.colorbar(fraction=0.1, pad=0.05)
    cbar.set_ticks(list(range(21)))
    cbar.set_ticklabels(st_AA)
    plt.ylabel('Sequences', size=ts)
    plt.xlabel('Position indices', size=ts)
    
    # Set y-axis limits to show only the sequence data (no gap)
    plt.ylim(0, n_samps)
    
    # Position text labels above the plot area using figure coordinates
    plt.text(23, n_samps * 1.05, "CDR1", transform=plt.gca().transData, ha='center')
    plt.text(48, n_samps * 1.05, "CDR2", transform=plt.gca().transData, ha='center')
    plt.text(111, n_samps * 1.05, "CDR3", transform=plt.gca().transData, ha='center')

    # Position horizontal lines just above the sequence data
    line_y = n_samps * 1.02
    plt.hlines(line_y, 0, max_seq_len-1, colors='cyan', lw=15, alpha=0.6, clip_on=False)
    plt.hlines(line_y, 26, 32, colors='magenta', lw=15, alpha=0.6, clip_on=False)
    plt.hlines(line_y, 48, 59, colors='magenta', lw=15, alpha=0.6, clip_on=False)
    plt.hlines(line_y, 101, 132, colors='magenta', lw=15, alpha=0.6, clip_on=False)
   
    plt.show()
    fig.tight_layout()
    fig.savefig(plot_fn, dpi=300)
    print(f"Plot saved as: {plot_fn}")