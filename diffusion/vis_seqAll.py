import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio import SeqIO
from src import GetOneHot, GetFasta 
from dotenv import load_dotenv
import os
from scipy.spatial.distance import jensenshannon

st_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU',
        'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
        'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
        'TYR', 'VAL', 'GAP']

one_let = [
   "A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", 
   "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"
]

def GetFreqs(seqs, one_let, n=21):
    ar_seq = np.array(seqs)
    frequencies = np.zeros(n)
    
    for i, el in enumerate(one_let):
        frequencies[i] = (ar_seq == el).sum()
    
    return frequencies

def total_variation_distance(p1, p2):
    return 0.5 * np.sum(np.abs(p1 - p2))

def calculate_distance_metrics(freqs1, freqs2):
    p1 = freqs1 / freqs1.sum()
    p2 = freqs2 / freqs2.sum()
    
    js_divergence = jensenshannon(p1, p2)
    tv_distance = total_variation_distance(p1, p2)
    
    return js_divergence, tv_distance, p1, p2

load_dotenv()
file2_path = os.getenv('FILE1_PATH')
file1_path = os.getenv('FILE2_PATH')

seq_list1, _ = GetFasta(file1_path)
seq_list2, _ = GetFasta(file2_path)

all_seq1 = []
for i, el in enumerate(seq_list1):
    all_seq1.extend(list(el))

all_seq2 = []
for i, el in enumerate(seq_list2):
    all_seq2.extend(list(el))

freqs1 = GetFreqs(all_seq1, one_let)
freqs2 = GetFreqs(all_seq2, one_let)
del all_seq1, all_seq2

freqs1 = freqs1[:-1]
freqs2 = freqs2[:-1]

js_div, tv_dist, p1, p2 = calculate_distance_metrics(freqs1, freqs2)

fig = plt.figure(figsize=(8, 5))
al = 0.8
n = 20

x = np.array(range(n))
width = 0.3

plt.bar(x - width/2, height=p1, width=width, alpha=al, label='Lowpoly Pool')
plt.bar(x + width/2, height=p2, width=width, alpha=al, label='Diffusion Generated')
plt.legend(frameon=False)

plt.xticks(range(n), st_AA[:-1], rotation=75)
plt.xlabel("Amino Acid")
plt.ylabel("Probability")
plt.title(f"Amino Acid Frequencies\nJS Divergence: {js_div:.4f}, TV Distance: {tv_dist:.4f}")
plt.tight_layout()

os.makedirs('./figs', exist_ok=True)

plt.savefig('./figs/all_freqs.png', dpi=300)
plt.show()