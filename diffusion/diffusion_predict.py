import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import tqdm
import time
from dotenv import load_dotenv
from diffusion_train import (
    CategoricalDiffusionTransformer, 
    CategoricalTransition, 
    load_model
)
from src import GetOneHot, GetFasta
from src import num_to_one
from src import write_multiple_fasta

def Sample(model, categorical_transition, seed, seq_len=146, 
           batch_size=5, num_steps=500, device='cpu', temperature=1):
    """    
    Sampler: Generate sequences using categorical diffusion
    """
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    model.eval()
    x_t = torch.randint(0, 21, (batch_size, seq_len), device=device)
    
    # Reverse diffusion process
    for t in tqdm.tqdm(reversed(range(num_steps)), desc=f"Seed {seed:3d}", leave=False):
        t_batch = torch.full((batch_size,), t, device=device)
        
        # Predict denoised probabilities
        pred_probs = model(x_t, t_batch)
        
        x_t = categorical_transition.denoise(x_t, pred_probs, t_batch, temperature)
    
    # Convert to probabilities for final output
    final_probs = model(x_t, torch.zeros(batch_size, device=device, dtype=torch.long))
    
    return x_t, final_probs

def GetOneLet(output):
    """Convert integer sequences to amino acid strings"""
    batch_size = output.shape[0]    
    gen_seq_list = []

    for i in range(batch_size):
        sequ = "".join([num_to_one[el] for el in output[i]])
        gen_seq_list.append(sequ)

    return gen_seq_list

if __name__=='__main__':
    load_dotenv()
    dat_path = os.getenv('DATA_PATH')
    file_name = os.getenv('FILE_NAME')
    if dat_path is None:
        raise ValueError("DATA_PATH environment variable is required")
    if file_name is None:
        raise ValueError("FILE_NAME environment variable is required")

    model_path = './checkpoints/categorical_diffusion_model_20250731_104046.pt'
    reference_fasta = os.path.join(dat_path, f'{file_name}.fasta')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 100       
    n_seeds = 140        
    out_path = 'output/'
    max_seq_len = 146    
    temperature = 1
    num_steps = 500      
    
    print(f"  - Device: {device}")
    
    os.makedirs(out_path + "sequences/", exist_ok=True)
    os.makedirs('visual/', exist_ok=True)
    
    print(f"\nLoading model from: {model_path}")
    model, categorical_transition, _, _ = load_model(model_path, device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    seq_list, _ = GetFasta(reference_fasta)
    print(f"Loaded {len(seq_list)} reference sequences")
    
    all_seq = []
    all_probs = []
    
    print(f"\n STARTING GENERATION")
    # Progress tracking
    completed_sequences = 0
    
    for i in range(n_seeds):
        output, probs = Sample(model,
                              categorical_transition,
                              seed=i,
                              seq_len=max_seq_len,
                              batch_size=batch_size,
                              num_steps=num_steps,
                              device=device,
                              temperature=temperature)

        # Convert to numpy for processing
        output = output.detach().cpu().numpy()     
        gen_seq_list = GetOneLet(output)
        all_seq = all_seq + gen_seq_list
        all_probs.append(probs.detach().cpu().numpy())
        completed_sequences += 1
        
        # Save intermediate results every 10 seeds
        if (i + 1) % 10 == 0:
            intermediate_fasta = out_path + f"sequences/Diffusion_gen_intermediate_{completed_sequences * batch_size}.fasta"
            write_multiple_fasta(all_seq, intermediate_fasta)
            print(f"Intermediate save: {len(all_seq)} sequences saved")
    
    print(f"\n GENERATION COMPLETE")
    
    # Save final sequences to FASTA
    fasta_output = out_path + "sequences/" + "Diffusion_gen_final.fasta"
    write_multiple_fasta(all_seq, fasta_output)
    print(f"Final sequences saved to: {fasta_output}")
    print(f"Total sequences generated: {len(all_seq)}")

    # Save probability distributions
    all_probs = np.concatenate(all_probs)
    np.save('visual/sequence_probabilities.npy', all_probs)
    print(f"Sequence probabilities saved to: visual/sequence_probabilities.npy")