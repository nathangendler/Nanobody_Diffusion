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
# Import your model classes and utilities
from diffusion_train import (
    CategoricalDiffusionTransformer, 
    CategoricalTransition, 
    compute_position_mask,
    load_model
)
from src import GetOneHot, GetFasta
from src import num_to_one
from src import write_multiple_fasta

def Sample(model, categorical_transition, position_mask, seed, seq_len=146, 
           batch_size=5, num_steps=500, device='cpu', temperature=0.8):
    """    
    Sampler: Generate sequences using categorical diffusion
    """
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    model.eval()
    x_t = torch.randint(0, 21, (batch_size, seq_len), device=device)
    position_mask_batch = position_mask.unsqueeze(0).expand(batch_size, -1)
    
    # Reverse diffusion process
    for t in tqdm.tqdm(reversed(range(num_steps)), desc=f"Seed {seed:3d}", leave=False):
        t_batch = torch.full((batch_size,), t, device=device)
        
        # Predict denoised probabilities
        pred_probs = model(x_t, t_batch, position_mask_batch, num_steps)
        
        x_t = categorical_transition.denoise(x_t, pred_probs, t_batch, temperature)
    
    # Convert to probabilities for final output
    final_probs = model(x_t, torch.zeros(batch_size, device=device, dtype=torch.long), 
                       position_mask_batch, num_steps)
    
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

    model_path = './final_models/categorical_diffusion_model_20250706_215530.pt'
    reference_fasta = os.path.join(dat_path, f'{file_name}.fasta')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 30       
    n_seeds = 100        
    out_path = 'output/'
    max_seq_len = 146    
    temperature = 0.9
    threshold = 0.2
    num_steps = 500      
    
    print(f"  - Device: {device}")
    
    
    # Create output directory
    os.makedirs(out_path + "sequences/", exist_ok=True)
    os.makedirs('visual/', exist_ok=True)
    
    ##########################################################################  
    ########## Load model
    print(f"\nLoading model from: {model_path}")
    model, categorical_transition, _, _ = load_model(model_path, device)
    model.eval()
    
    print(f"Model loaded successfully")
    
    seq_list, _ = GetFasta(reference_fasta)
    
    
    # Convert to one-hot for position mask computation
    OHot = GetOneHot(seq_list, max_seq_len, device=device)
    
    # Compute position mask (0=framework, 1=variable)
    position_mask, variability = compute_position_mask(OHot, threshold=threshold)
    
    framework_positions = (position_mask == 0).sum().item()
    variable_positions = (position_mask == 1).sum().item()
    
    print(f"Position analysis:")
    print(f"  Framework positions: {framework_positions}")
    print(f"  Variable positions: {variable_positions}")
    
    ###########################################################################
    # Generate sequences
    all_seq = []
    all_probs = []
    
    print(f"\n STARTING GENERATION")
    # Progress tracking
    completed_sequences = 0
    
    for i in range(n_seeds):
        output, probs = Sample(model,
                              categorical_transition,
                              position_mask,
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
        
        # Save intermediate results every 50 seeds (250 sequences)
        if (i + 1) % 10 == 0:
            intermediate_fasta = out_path + f"sequences/Diffusion_gen_intermediate_0.9_3000.fasta"
            write_multiple_fasta(all_seq, intermediate_fasta)
            print(f"Intermediate save: {completed_sequences} sequences saved")
    
    print(f"\n GENERATION COMPLETE")
    
    # Save final sequences to FASTA
    fasta_output = out_path + "sequences/" + "Diffusion_gen_0.9__final.fasta"
    write_multiple_fasta(all_seq, fasta_output)
    print(f"Final sequences saved to: {fasta_output}")

    # Save probability distributions
    all_probs = np.concatenate(all_probs)
    np.save('visual/sequence_probabilities_1000.npy', all_probs)
    np.save('visual/position_mask.npy', position_mask.cpu().numpy())
    np.save('visual/position_variability.npy', variability.cpu().numpy())
