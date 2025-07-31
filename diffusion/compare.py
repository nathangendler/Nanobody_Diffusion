#!/usr/bin/env python3
"""
Nanobody Sequence Comparison Script
Compares sequences from two FASTA files to find matches and similarities
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def read_fasta_file(file_path):
    """Read sequences from a FASTA file"""
    sequences = {}
    current_id = None
    current_seq = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    
                    # Start new sequence
                    current_id = line[1:]  # Remove '>' character
                    current_seq = []
                else:
                    # Add to current sequence
                    current_seq.append(line)
            
            # Don't forget the last sequence
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
                
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)
    
    return sequences

def clean_sequence(seq):
    """Remove gaps and formatting characters from sequence"""
    return seq.replace('-', '').replace(' ', '').replace('\n', '').upper()

def find_exact_matches(group1_seqs, group2_seqs):
    """Find exact matches between two groups of sequences"""
    matches = []
    
    for id1, seq1 in group1_seqs.items():
        clean_seq1 = clean_sequence(seq1)
        for id2, seq2 in group2_seqs.items():
            clean_seq2 = clean_sequence(seq2)
            if clean_seq1 == clean_seq2 and len(clean_seq1) > 0:
                matches.append((id1, id2, clean_seq1))
    
    return matches

def calculate_similarity(seq1, seq2):
    """Calculate percentage similarity between two sequences"""
    seq1_clean = clean_sequence(seq1)
    seq2_clean = clean_sequence(seq2)
    
    if len(seq1_clean) == 0 or len(seq2_clean) == 0:
        return 0.0
    
    # Use the longer sequence as the base for percentage calculation
    max_len = max(len(seq1_clean), len(seq2_clean))
    min_len = min(len(seq1_clean), len(seq2_clean))
    
    # Count matching positions
    matches = 0
    for i in range(min_len):
        if seq1_clean[i] == seq2_clean[i]:
            matches += 1
    
    # Calculate similarity as percentage
    similarity = (matches / max_len) * 100
    return similarity

def find_similar_sequences(group1_seqs, group2_seqs, threshold=80):
    """Find sequences with similarity above threshold"""
    similar_pairs = []
    
    for id1, seq1 in group1_seqs.items():
        for id2, seq2 in group2_seqs.items():
            similarity = calculate_similarity(seq1, seq2)
            if similarity >= threshold:
                similar_pairs.append((id1, id2, similarity, clean_sequence(seq1), clean_sequence(seq2)))
    
    return similar_pairs

def get_file_paths():
    """Get file paths from environment variables or command line arguments"""
    # Try environment variables first
    generated_sequences = os.getenv('GENERATED_SEQUENCES')
    file1_path = os.getenv('FILE1_PATH')
    
    # Debug: print what we found
    print(f"Debug - GENERATED_SEQUENCES env var: {generated_sequences}")
    print(f"Debug - FILE1_PATH env var: {file1_path}")
    
    # If not found in env vars, try command line arguments
    if not generated_sequences or not file1_path:
        if len(sys.argv) >= 3:
            generated_sequences = sys.argv[1]
            file1_path = sys.argv[2]
        else:
            print("Usage:")
            print("  Set environment variables:")
            print("    GENERATED_SEQUENCES=path/to/generated/sequences.fasta")
            print("    FILE1_PATH=path/to/reference/sequences.fasta")
            print("  OR pass as command line arguments:")
            print("    python script.py <generated_sequences.fasta> <reference_sequences.fasta>")
            sys.exit(1)
    
    # Validate file paths
    if not Path(generated_sequences).exists():
        print(f"Error: Generated sequences file not found: {generated_sequences}")
        sys.exit(1)
    
    if not Path(file1_path).exists():
        print(f"Error: Reference sequences file not found: {file1_path}")
        sys.exit(1)
    
    return generated_sequences, file1_path

def main():
    """Main function to run the sequence comparison"""
    print("=== Nanobody Sequence Comparison ===\n")
    
    # Get file paths
    generated_sequences_path, reference_sequences_path = get_file_paths()
    
    print(f"Generated sequences: {generated_sequences_path}")
    print(f"Reference sequences: {reference_sequences_path}")
    print()
    
    # Read sequences from both files
    print("Reading sequences...")
    generated_sequences = read_fasta_file(generated_sequences_path)
    reference_sequences = read_fasta_file(reference_sequences_path)
    
    print(f"Generated sequences: {len(generated_sequences)} sequences")
    print(f"Reference sequences: {len(reference_sequences)} sequences")
    print()
    
    # Show sample sequence info
    if generated_sequences:
        first_gen_id = list(generated_sequences.keys())[0]
        first_gen_seq = clean_sequence(generated_sequences[first_gen_id])
        print(f"Sample generated sequence ({first_gen_id}): {len(first_gen_seq)} amino acids")
        print(f"  {first_gen_seq[:60]}...")
    
    if reference_sequences:
        first_ref_id = list(reference_sequences.keys())[0]
        first_ref_seq = clean_sequence(reference_sequences[first_ref_id])
        print(f"Sample reference sequence ({first_ref_id}): {len(first_ref_seq)} amino acids")
        print(f"  {first_ref_seq[:60]}...")
    
    print("\n" + "="*60)
    
    # Find exact matches
    print("\n=== EXACT MATCHES ===")
    exact_matches = find_exact_matches(generated_sequences, reference_sequences)
    
    if exact_matches:
        print(f"Found {len(exact_matches)} exact match(es):")
        for i, match in enumerate(exact_matches, 1):
            print(f"\n{i}. EXACT MATCH:")
            print(f"   Generated: {match[0]}")
            print(f"   Reference: {match[1]}")
            print(f"   Length: {len(match[2])} amino acids")
            print(f"   Sequence: {match[2][:80]}...")
    else:
        print("No exact matches found.")
    
    # Find high similarity matches (95% threshold)
    print(f"\n=== HIGH SIMILARITY MATCHES (≥95%) ===")
    high_similarity = find_similar_sequences(generated_sequences, reference_sequences, 95)
    
    if high_similarity:
        print(f"Found {len(high_similarity)} high similarity match(es):")
        for i, match in enumerate(high_similarity, 1):
            print(f"\n{i}. HIGH SIMILARITY ({match[2]:.1f}%):")
            print(f"   Generated: {match[0]}")
            print(f"   Reference: {match[1]}")
            print(f"   Gen length: {len(match[3])}, Ref length: {len(match[4])}")
    else:
        print("No high similarity matches found.")
    
    # Find moderate similarity matches (80% threshold)
    print(f"\n=== MODERATE SIMILARITY MATCHES (≥80%) ===")
    moderate_similarity = find_similar_sequences(generated_sequences, reference_sequences, 80)
    moderate_similarity = [m for m in moderate_similarity if m[2] < 95]  # Exclude high similarity ones
    
    if moderate_similarity:
        print(f"Found {len(moderate_similarity)} moderate similarity match(es):")
        for i, match in enumerate(moderate_similarity[:10], 1):  # Show top 10
            print(f"{i}. {match[2]:.1f}% - {match[0]} vs {match[1]}")
        if len(moderate_similarity) > 10:
            print(f"   ... and {len(moderate_similarity) - 10} more")
    else:
        print("No moderate similarity matches found.")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    all_comparisons = find_similar_sequences(generated_sequences, reference_sequences, 0)
    all_comparisons.sort(key=lambda x: x[2], reverse=True)
    
    if all_comparisons:
        max_similarity = all_comparisons[0][2]
        avg_similarity = sum(comp[2] for comp in all_comparisons) / len(all_comparisons)
        
        print(f"Total comparisons: {len(all_comparisons)}")
        print(f"Highest similarity: {max_similarity:.1f}%")
        print(f"Average similarity: {avg_similarity:.1f}%")
        
        # Count by similarity ranges
        exact_count = len([c for c in all_comparisons if c[2] == 100])
        high_count = len([c for c in all_comparisons if 95 <= c[2] < 100])
        moderate_count = len([c for c in all_comparisons if 80 <= c[2] < 95])
        low_count = len([c for c in all_comparisons if c[2] < 80])
        
        print(f"\nSimilarity distribution:")
        print(f"  Exact matches (100%): {exact_count}")
        print(f"  High similarity (95-99%): {high_count}")
        print(f"  Moderate similarity (80-94%): {moderate_count}")
        print(f"  Low similarity (<80%): {low_count}")
        
        # Show top 5 matches overall
        print(f"\nTop 5 most similar pairs:")
        for i, comp in enumerate(all_comparisons[:5], 1):
            print(f"  {i}. {comp[2]:.1f}% - {comp[0]} vs {comp[1]}")

if __name__ == "__main__":
    main()