import numpy as np
import io
import os
import torch.nn.functional as F
import torch
from Bio import SeqIO
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


num_to_one = {0: "A",  1: "R",  2: "N",  3: "D",  4: "C",
              5: "E",  6: "Q",  7: "G",  8: "H",  9: "I",
              10: "L", 11: "K", 12: "M", 13: "F", 14: "P",
              15: "S", 16: "T", 17: "W", 18: "Y", 19: "V",
              20: '-'}      


one_to_num = {
    "A": 0,   # Alanine
    "R": 1,   # Arginine
    "N": 2,   # Asparagine
    "D": 3,   # Aspartic acid
    "C": 4,   # Cysteine
    "E": 5,   # Glutamic acid
    "Q": 6,   # Glutamine
    "G": 7,   # Glycine
    "H": 8,   # Histidine
    "I": 9,   # Isoleucine
    "L": 10,  # Leucine
    "K": 11,  # Lysine
    "M": 12,  # Methionine
    "F": 13,  # Phenylalanine
    "P": 14,  # Proline
    "S": 15,  # Serine
    "T": 16,  # Threonine
    "W": 17,  # Tryptophan
    "Y": 18,  # Tyrosine
    "V": 19,  # Valine
    "-": 20   # Gap
}




def GetFasta(fasta_file):
    """
    
    """
    
    one_letter_sequences = []
    len_list = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        one_letter_seq = str(record.seq)
        one_letter_sequences.append(one_letter_seq)
        len_list.append(len(one_letter_seq))
        
    return one_letter_sequences, len_list



def GetOneHot(seq_list, max_seq_len, device='cpu'):
    """ 
    Get one hot encoding from AA sequence.

    Input:
        seq_list - sequence list per protein, 3 letter AA code
        max_seq_leng - max sequence length, set to size of longest sequnce
        
    Output:
        res_batch - one hot encoded and padded sequence encodings
        seq_mask  - mask that has ones in positions of actuall AAs and zeros in positions of pads
                    This will be used with loss function
                    
    """
    
    batch_size = len(seq_list)
    one_hot_batch = torch.zeros(batch_size, max_seq_len, 21, device = device, dtype = torch.float)
    
    
    i = 0
    for seq in seq_list:
        seq = torch.as_tensor([one_to_num[a] for a in seq], device=device, dtype=torch.long)        
        seqOneHot = F.one_hot(seq, 21)
        
        one_hot_batch[i,:,:] = seqOneHot
        i += 1
    return one_hot_batch

def write_multiple_fasta(sequences, output_file):
    """
    Writes multiple 1-letter amino acid sequences to a FASTA file.

    :param sequences: list of 1-letter amino acid sequences
    :param output_file: Name of the output FASTA file
    """
    seq_records = []
    i = 0
    for sequence in sequences:
        seq_record = SeqRecord(
            Seq(sequence),  # Convert sequence string to Biopython Seq object
            id = "genAI nanobody " + str(i) ,  
            description=f"Protein sequence"  # Custom description
        )
        seq_records.append(seq_record)
        i += 1
        
    # Write all sequences to the FASTA file
    with open(output_file, "w") as fasta_out:
        SeqIO.write(seq_records, fasta_out, "fasta")
        
def GetNumeric(seqs, max_seq_len, device='cpu'):
    """ 
    Get numeric encoding from AA sequence.
                    
    """
    
    n_seqs = len(seqs)
    
    one_hot_batch = torch.zeros(n_seqs, max_seq_len, device = device, dtype = torch.long) + 20
  
    i = 0
    for seq in seqs:
        num_seq = torch.as_tensor([one_to_num[a] for a in seq], device=device, dtype=torch.long)        
       
        # zero pad sequences
        one_hot_batch[i,:] = num_seq 
        i += 1
    
    return one_hot_batch


if __name__ == "__main__":

    
    fasta_path = '../toy_data/'
    file_name = "aligned_2000"
    
    max_seq_len = 144    
    seq_list, len_list = GetFasta(fasta_path + file_name + '.fasta')
    one_hot_batch = GetOneHot(seq_list, max_seq_len, device='cuda')



