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
#typ = 'recon_test'
#typ = 'test'
#typ = 'test_biased'
#typ = 'gen'
typ = 'gen_around_WT'

load_dotenv()
file_name = os.getenv('GENERATED_SEQUENCES')

device = 'cpu'
max_seq_len = 146
seq_list, len_list = GetFasta(file_name)

one_hot_batch = GetOneHot(seq_list, max_seq_len, device='cpu')
numeric_batch = GetNumeric(seq_list, max_seq_len, device='cpu')
n_samps = one_hot_batch.shape[0]


#ytick_positions = list(range(0,80000, 10000))
#ytick_labels = [ str(el) + "K" for el in range(0,80,10) ]


if plot:
    cmap = "tab20c"
    lc = 'darkblue'
    ts = 13
    plot_fn = 'Sequences_aligned_' + typ + '.png'
    
    fig = plt.figure(figsize=[8,3.5])

    #plt.title(typ)
    plt.imshow(numeric_batch, aspect=.05, interpolation=None, origin='lower',
               cmap=cmap, alpha=0.8)

    cbar = plt.colorbar(fraction=0.1, pad=0.05)
    cbar.set_ticks(list(range(21)))
    cbar.set_ticklabels(st_AA)
    #plt.yticks(ytick_positions, ytick_labels, rotation=0, size=ts-2)
    plt.ylabel('Sequences', size=ts)
    plt.xlabel('Position indecies', size=ts)
    
    # Set y-axis limits to show only the sequence data (no gap)
    plt.ylim(0, n_samps)
    
    # Position text labels above the plot area using figure coordinates
    # This places them outside the data area but visible
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
    #fig.tight_layout()
    #os.system("epscrop %s %s" % (plot_fn, plot_fn))
    #fig.savefig(plot_fn, dpi=300)