## File descriptions

- train.py:    script to train the model
- predict.py:  samples new sequences from gaussian noise in the latent space
- recon_test:  reconstructs the test set, so we can estimate model reconstruction accuracy
- LinInerpol:  performs linear interpolation between two ref points in the latent space
- 



## Parameters and Desicions

1. Here we use Sergios aligment as input seqs
   * One hot encoding done on 21 letters
   * 21th letter (20) is a gap, model also learns gap positions
 
2. We also used low latent dimension, 64 .

3. Small min penalty used per latent point, 0.5 nats (free bits)

4. Included rel_pos embeddings, it is supposed to help with translation invariance!     

5. Reconstruction accuracy ~90%, fine for VAE

6. Generated sequences resemble original, more investigation needed

7. Latent space interpolation is smooth, distance has meaning, close points correspond to similar sequence




✅ A. When Can You Trust the Model?

You can trust your VAE when:

    ✅ Reconstruction is qualitatively meaningful (not perfect, but coherent).

    ✅ The latent space is smooth, and interpolations between samples are valid.

    ✅ The KL divergence is non-zero, meaning the latent space is used.

    ✅ Sampling from the latent space yields diverse, valid outputs.

    ✅ Downstream tasks (e.g., classification, clustering, generation) perform well using 
