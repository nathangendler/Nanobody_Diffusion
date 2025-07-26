# Nanobody Diffusion: Amino Acid Sequence Generation using Diffusion Models

## A research project for nanobody amino acid sequence generation with diffusion models in Python using Pytorch

This project demonstrates how to use diffusion models for generating nanobody (single-domain antibody) amino acid sequences. It is designed as a research resource, providing a full pipeline from data preparation to model training, prediction, and visualization. Key features in this project include:

- Preparing and processing amino acid sequence data for machine learning
- Training a diffusion model to generate nanobody sequence generation
- Create new nanobody amino acid sequences from random data using model
- Script to visualize sequence data and model outputs


## Results

![Amino Acid Frequency](/diffusion/AAfrequency.png)
<br>
<sub><i>*Amino Acid Frequency of generated sequence (orange) vs test set (blue)*</i></sub>

## How to install and run the Nanobody Diffusion project

The easiest way to get started is to follow these steps:

1. **Clone this project**
   ```bash
   git clone https://github.com/nathangendler/Nanobody_Diffusion
   cd Nanobody_Diffusion
   ```

2. **Set up your Python environment**
   - Install Python 3.8+ (recommend using [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
   - Create and activate a new environment:
     ```bash
     conda create -n nanobody_diffusion python=3.8
     conda activate nanobody_diffusion
     ```

3. **Install dependencies**
    - Dependencies can be found in the requirements.txt file
        ```bash
        pip install -r requirements.txt
        ```

4. **Prepare your data**
   - Place your amino acid sequence data (e.g., FASTA files) in the `data/` directory and edit the path in diffusion/diffusion_train.py

5. **Train the diffusion model**
   - Run the training script:
     ```bash
     python diffusion/diffusion_train.py
     ```

6. **Predict new sequences**
   - Use the prediction script:
     ```bash
     python diffusion/diffusion_predict.py
     ```

7. **Visualize results**
   - Explore the `output/` and `diffusion/visual/` directories for visualization tools and results.

## Future edits and modifications

This project is intended as a starting point for amino acid sequence generation research. You are encouraged to fork, clone, and modify the codebase for your own experiments. The modular structure makes it easy to:

- Adjust model architectures in `diffusion/src/`
- Extend training and prediction scripts
- Integrate new visualization tools
