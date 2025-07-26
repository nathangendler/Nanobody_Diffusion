import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio import SeqIO
from src import GetOneHot, GetFasta 
from dotenv import load_dotenv
import os

st_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU',
        'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
        'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
        'TYR', 'VAL', 'GAP']

one_let = [
   "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
   "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"
]

def GetFreqs(seqs, one_let, n=21):
    """Calculate amino acid frequencies"""
    print(f"Processing {len(seqs)} amino acids...")
    ar_seq = np.array(seqs)
    frequencies = np.zeros(n)
    
    for i, el in enumerate(one_let):
        frequencies[i] = (ar_seq == el).sum()
        if i % 5 == 0:  # Progress indicator
            print(f"Processed {i+1}/{n} amino acids")
    
    return frequencies
load_dotenv()
# File paths
file2_path = os.getenv('FILE1_PATH')
file1_path = os.getenv('FILE2_PATH')

print("Loading file 1")
seq_list1, _ = GetFasta(file1_path)
print(f"File 1: {len(seq_list1)} sequences loaded")

print("Loading file 2")
seq_list2, _ = GetFasta(file2_path)
print(f"File 2: {len(seq_list2)} sequences loaded")

print(f"Sample sequence lengths from file 1: {[len(seq) for seq in seq_list1[:5]]}")
print(f"Sample sequence lengths from file 2: {[len(seq) for seq in seq_list2[:5]]}")

all_seq1 = []
for i, el in enumerate(seq_list1):
    all_seq1.extend(list(el))

print(f"Total amino acids in file 1: {len(all_seq1)}")

all_seq2 = []
for i, el in enumerate(seq_list2):
    all_seq2.extend(list(el))

freqs1 = GetFreqs(all_seq1, one_let)
freqs2 = GetFreqs(all_seq2, one_let)
del all_seq1, all_seq2

freqs1 = freqs1[:-1]
freqs2 = freqs2[:-1]

fig = plt.figure(figsize=(6, 4))
al = 0.8
n = 20

x = np.array(range(n))
width = 0.3

plt.bar(x - width/2, height=freqs1/freqs1.sum(), width=width, alpha=al, label='Lowpoly Pool')
plt.bar(x + width/2, height=freqs2/freqs2.sum(), width=width, alpha=al, label='Diffusion Generated')
plt.legend(frameon=False)

plt.xticks(range(n), st_AA[:-1], rotation=75)
plt.xlabel("Amino Acid")
plt.ylabel("P")
plt.title("Amino Acid Frequencies for Nanobody Sequences")
plt.tight_layout()

# Make sure the figs directory exists
import os
os.makedirs('./figs', exist_ok=True)

plt.savefig('./figs/all_freqs.png', dpi=300)
print("Plot saved successfully")
plt.show()