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
    """Calculate amino acid frequencies"""
    print(f"Processing {len(seqs)} amino acids...")
    ar_seq = np.array(seqs)
    frequencies = np.zeros(n)
    
    for i, el in enumerate(one_let):
        frequencies[i] = (ar_seq == el).sum()
        if i % 5 == 0:  # Progress indicator
            print(f"Processed {i+1}/{n} amino acids")
    
    return frequencies

def total_variation_distance(p1, p2):
    """Calculate Total Variation Distance between two probability distributions"""
    return 0.5 * np.sum(np.abs(p1 - p2))

def calculate_distance_metrics(freqs1, freqs2):
    """Calculate both Jensen-Shannon divergence and Total Variation distance"""
    # Normalize to probabilities
    p1 = freqs1 / freqs1.sum()
    p2 = freqs2 / freqs2.sum()
    
    # Calculate metrics
    js_divergence = jensenshannon(p1, p2)
    tv_distance = total_variation_distance(p1, p2)
    
    return js_divergence, tv_distance, p1, p2

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

# Remove gap frequencies for analysis
freqs1 = freqs1[:-1]
freqs2 = freqs2[:-1]

# Calculate distance metrics
js_div, tv_dist, p1, p2 = calculate_distance_metrics(freqs1, freqs2)

# Print results
print("\n" + "="*50)
print("AMINO ACID FREQUENCY COMPARISON RESULTS")
print("="*50)
print(f"Jensen-Shannon Divergence: {js_div:.4f}")
print(f"Total Variation Distance:  {tv_dist:.4f}")
print()

# Interpretation
if js_div < 0.1:
    js_interp = "Very similar distributions"
elif js_div < 0.3:
    js_interp = "Moderately different distributions"
else:
    js_interp = "Very different distributions"

if tv_dist < 0.05:
    tv_interp = "Very similar (≤5% difference)"
elif tv_dist < 0.15:
    tv_interp = "Moderately different (5-15% difference)"
else:
    tv_interp = "Very different (>15% difference)"

print(f"JS Interpretation: {js_interp}")
print(f"TV Interpretation: {tv_interp}")
print("="*50)

# Find amino acids with largest differences
abs_diff = np.abs(p1 - p2)
top_diff_indices = np.argsort(abs_diff)[-5:][::-1]  # Top 5 differences

print("\nTop 5 Amino Acids with Largest Frequency Differences:")
for i, idx in enumerate(top_diff_indices):
    aa = st_AA[idx]
    diff = abs_diff[idx]
    freq1 = p1[idx]
    freq2 = p2[idx]
    print(f"{i+1}. {aa}: |{freq1:.3f} - {freq2:.3f}| = {diff:.3f}")

# Create the plot
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

# Make sure the figs directory exists
os.makedirs('./figs', exist_ok=True)

plt.savefig('./figs/all_freqs.png', dpi=300)
print("\nPlot saved successfully to ./figs/all_freqs.png")
plt.show()