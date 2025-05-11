import os
import time
from trainer_PPO_trXL import make_train
import wandb
import jax
import jax.numpy as jnp

# Set device in a safer way
try:
    # Try to print available devices to help with debugging
    print("Available devices:", jax.devices())
except:
    # If that fails, just set CPU platform directly
    print("Error listing devices, forcing CPU")
    os.environ["JAX_PLATFORMS"] = "cpu"

config = {
    "LR": 2e-4,
    "NUM_ENVS": 512,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 1e9,
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
    "WANDB_MODE": "online",  # Set to "online" to enable wandb logging
    "WANDB_PROJECT": "lit-transformer-ppo",
    "WANDB_ENTITY": "maharishiva",  # Set to your wandb username or team name
    "WANDB_LOG_FREQ": 1,    # Log every N updates
    "THINKING_VOCAB": 64,
    "R_THINK": 0.0,
    "MAX_THINKING_LEN": 8,
}

# Initialize wandb if enabled
if config["WANDB_MODE"] == "online":
    wandb.init(
        project=config["WANDB_PROJECT"],
        entity=config["WANDB_ENTITY"],
        config=config,
        name=f"{config['ENV_NAME']}_seed{config['seed']}",
    )

seed=config["seed"]

prefix = "results_craftax/"+config["ENV_NAME"]

# Create directories with proper error handling
try:
    if not os.path.exists("results_craftax"):
        os.makedirs("results_craftax")
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    print(f"Saving results to {prefix}")
except Exception as e:
    print(f"Directory creation {prefix} failed: {str(e)}")
    
print("Start compiling and training")
time_a=time.time()
rng = jax.random.PRNGKey(seed)

train_jit = jax.jit(make_train(config))
print(f"Compilation finished in {time.time() - time_a:.2f} seconds")

out = train_jit(rng)
print("training and compilation took " + str(time.time()-time_a))

# Close wandb run if it was enabled
if config["WANDB_MODE"] == "online":
    wandb.finish()

import matplotlib.pyplot as plt
plt.plot(out["metrics"]["returned_episode_returns"])
plt.xlabel("Updates")
plt.ylabel("Return")
plt.savefig(prefix+"/return_"+str(seed))

plt.clf()

jnp.save(prefix+"/"+str(seed)+"_params", out["runner_state"][0].params)
jnp.save(prefix+"/"+str(seed)+"_config", config)

jnp.save(prefix+"/"+str(seed)+"_metrics",out["metrics"])
