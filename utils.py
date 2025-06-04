import os, random, numpy as np, torch

def set_seed():
    """Set random seed for reproducibility."""
    SEED = 42                               # pick any integer you like

    os.environ["PYTHONHASHSEED"] = str(SEED)     # hash randomisation
    random.seed(SEED)                            # built-in RNG
    np.random.seed(SEED)                         # NumPy RNG
    torch.manual_seed(SEED)                      # CPU & default CUDA RNG
    torch.cuda.manual_seed_all(SEED)             # all GPUs