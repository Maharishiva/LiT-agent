"""
# Running Transformer-XL PPO on Google Colab

This file contains instructions for setting up and running the Transformer-XL PPO project on Google Colab.
Copy and paste these code blocks into a Colab notebook.

## 1. Clone Repository

```python
# Clone the repository
!git clone https://github.com/Maharishiva/LiT-agent.git
%cd LiT-agent
```

## 2. Install Dependencies

```python
# Install JAX with GPU support first
!pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install all requirements from requirements.txt
!pip install -r requirements.txt
```

## 3. Set Up Weights & Biases (wandb)

```python
# Log in to wandb - you'll need to provide your API key
!wandb login
```

## 4. Run Training

```python
# Run with default settings
!python train_PPO_trXL.py
```

## 5. Alternative: Modify Configuration

If you want to modify the configuration for Colab (e.g., reduce NUM_ENVS to save memory):

```python
import train_PPO_trXL
import jax

# Display available devices
print("Available devices:", jax.devices())

# Modify config as needed
config = {
    "LR": 2e-4,
    "NUM_ENVS": 128,  # Reduced from 512 for Colab
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 5e5,  # Reduced for testing
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.8,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 1.,
    "ACTIVATION": "relu",
    "ENV_NAME": "craftax",
    "ANNEAL_LR": True,
    "qkv_features": 256,
    "EMBED_SIZE": 256,
    "num_heads": 8,
    "num_layers": 2,
    "hidden_layers": 256,
    "WINDOW_MEM": 128,
    "WINDOW_GRAD": 64,
    "gating": True,
    "gating_bias": 2.,
    "seed": 0,
    "WANDB_MODE": "online",
    "WANDB_PROJECT": "lit-transformer-ppo",
    "WANDB_ENTITY": "maharishiva",
    "WANDB_LOG_FREQ": 10,
}

# Run training with modified config
rng = jax.random.PRNGKey(config["seed"])
train_jit = jax.jit(train_PPO_trXL.make_train(config))
out = train_jit(rng)
```

## 6. Monitor Memory Usage

```python
# Check GPU memory usage
!nvidia-smi
```

## 7. Keep Colab Session Active

```python
# Enable custom widget manager to help prevent disconnections
from google.colab import output
output.enable_custom_widget_manager()
```

## Important Notes:

1. Make sure to select GPU runtime in Colab (Runtime → Change runtime type → GPU)
2. If you encounter memory errors, try reducing NUM_ENVS in the configuration
3. Colab sessions have time limits (~12 hours max), so use wandb to track progress
4. For longer training runs, consider implementing checkpointing to save progress
""" 