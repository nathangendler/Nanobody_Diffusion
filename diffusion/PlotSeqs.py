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

n_samps = numeric_batch.shape[0]

if plot:
    cmap = "tab20c"
    lc = 'darkblue'
    ts = 13
    plot_fn = 'Sequences_aligned_' + typ + f'_first_{n_samps}.png'
    
    fig = plt.figure(figsize=[8,3.5])

    plt.imshow(numeric_batch, aspect='auto', interpolation=None, origin='lower',
               cmap=cmap, alpha=0.8)

    cbar = plt.colorbar(fraction=0.1, pad=0.05)
    cbar.set_ticks(list(range(21)))
    cbar.set_ticklabels(st_AA)
    plt.ylabel('Sequences', size=ts)
    plt.xlabel('Position indices', size=ts)
    
    plt.ylim(0, n_samps)
    
    plt.text(23, n_samps * 1.05, "CDR1", transform=plt.gca().transData, ha='center')
    plt.text(48, n_samps * 1.05, "CDR2", transform=plt.gca().transData, ha='center')
    plt.text(111, n_samps * 1.05, "CDR3", transform=plt.gca().transData, ha='center')

    line_y = n_samps * 1.02
    plt.hlines(line_y, 0, max_seq_len-1, colors='cyan', lw=15, alpha=0.6, clip_on=False)
    plt.hlines(line_y, 26, 32, colors='magenta', lw=15, alpha=0.6, clip_on=False)
    plt.hlines(line_y, 48, 59, colors='magenta', lw=15, alpha=0.6, clip_on=False)
    plt.hlines(line_y, 101, 132, colors='magenta', lw=15, alpha=0.6, clip_on=False)
   
    plt.show()
    fig.tight_layout()
    fig.savefig(plot_fn, dpi=300)