import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio import SeqIO
from Fasta import GetFasta, GetOneHot, GetNumeric
import torch
import os


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


file_name = file_path + 'VAE_' + typ

device = 'cpu'
max_seq_len = 146
seq_list, len_list = GetFasta(file_name + '.fasta')

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
    #plt.imshow(numeric_batch, aspect=.15, interpolation=None, origin='lower',
    #           cmap=cmap, alpha=0.8)
    plt.imshow(numeric_batch, aspect=.005, interpolation=None, origin='lower',
               cmap=cmap, alpha=0.8)

    cbar = plt.colorbar(fraction=0.1, pad=0.05)
    cbar.set_ticks(list(range(21)))
    cbar.set_ticklabels(st_AA)
    #plt.yticks(ytick_positions, ytick_labels, rotation=0, size=ts-2)
    plt.ylabel('Sequences', size=ts);  plt.xlabel('Position indecies', size=ts)
    
    plt.text(23, n_samps+ 1500, "CDR1")
    plt.text(48, n_samps+ 1500, "CDR2")
    plt.text(111, n_samps+ 1500, "CDR3")

    ypos = 500
    plt.hlines(n_samps+ypos, 0, max_seq_len-1, colors='cyan', lw=15, alpha=0.6)
    plt.hlines(n_samps+ypos, 26, 32, colors='magenta', lw=15 , alpha=0.6)
    plt.hlines(n_samps+ypos, 48, 59, colors='magenta', lw=15, alpha=0.6)
    plt.hlines(n_samps+ypos, 101, 132, colors='magenta', lw=15, alpha=0.6)
   

    fig.set_tight_layout(True)
    os.system("epscrop %s %s" % (plot_fn, plot_fn))
    fig.savefig(plot_fn, dpi=300)