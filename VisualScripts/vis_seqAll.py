import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio import SeqIO
from Fasta import GetFasta

st_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU',
         'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
         'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
         'TYR', 'VAL', 'GAP']

one_let = [
    "A",  # Alanine
    "R",  # Arginine
    "N",  # Asparagine
    "D",  # Aspartic acid
    "C",  # Cysteine
    "E",  # Glutamic acid
    "Q",  # Glutamine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "L",  # Leucine
    "K",  # Lysine
    "M",  # Methionine
    "F",  # Phenylalanine
    "P",  # Proline
    "S",  # Serine
    "T",  # Threonine
    "W",  # Tryptophan
    "Y",  # Tyrosine
    "V",  # Valine
    "-"   # Gap
    ]


def GetFreqs(seqs, one_let, n=21):
    """
    
    """
    # Example: Visualizing amino acid frequencies
    ar_seq = np.array(seqs)
    frequencies = np.zeros(n)

    i = 0
    for el in one_let:
        frequencies[i] = (ar_seq == el).sum()
        i += 1
    
    return frequencies



file_path = '../'
f_type1 = 'recon_test'
f_type2 = 'test'
f_type3 = 'gen'

file_name1 = 'VAE_' + f_type1
file_name2 = 'VAE_' + f_type2
file_name3 = 'VAE_' + f_type3


device = 'cpu'
max_seq_len = 146

seq_list1, _ = GetFasta(file_path + file_name1 + '.fasta')
seq_list2, _ = GetFasta(file_path + file_name2 + '.fasta')
seq_list3, _ = GetFasta(file_path + file_name3 + '.fasta')

all_seq1 = []; all_seq2 = [] ; all_seq3 = []

for el in seq_list1:
    all_seq1 = all_seq1 + list(el)
    
for el in seq_list2:
    all_seq2 = all_seq2 + list(el)

for el in seq_list3:
    all_seq3 = all_seq3 + list(el)

 
freqs1 = GetFreqs(all_seq1, one_let)
freqs2 = GetFreqs(all_seq2, one_let)
freqs3 = GetFreqs(all_seq3, one_let)

del all_seq1, all_seq2, all_seq3
amino_acids = np.array(st_AA)

 
freqs1 = freqs1[:-1]
freqs2 = freqs2[:-1]
freqs3 = freqs3[:-1]

#freqs1[8] = freqs1[8] + 2000 #?
#freqs3[8] = freqs3[8] + 2000 #?


## plot ##################################3
fig = plt.figure(figsize=(6, 4))
al = 0.8
n = 20

x = np.array(range(n))
width = 0.2

plt.bar(x - width, height = freqs1/freqs1.sum(), width = width, alpha=al)
plt.bar(x , height = freqs2/freqs2.sum(), width = width, alpha=al)
plt.bar(x + width, height = freqs3/freqs3.sum(), width = width, alpha=al)
plt.legend(['test set', 'reconstructed', 'generated'], frameon=False)

plt.xticks(range(n), st_AA[:-1], rotation=75)
plt.xlabel("Amino Acid")
plt.ylabel("P")
plt.title("Amino Acid Frequencies for Nanobody Sequences")
fig.set_tight_layout(True)
plt.savefig('./figs/' + "all_freqs.png", dpi=300)
plt.show()