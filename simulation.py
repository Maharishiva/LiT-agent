import jax
import jax.numpy as jnp
import numpy as np
import imageio
from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
from craftax.craftax.renderer import render_craftax_symbolic
from transformerXL import Transformer
from trainer_PPO_trXL import ActorCriticTransformer

# Load saved model parameters and config
params = jnp.load("results_craftax/craftax/0_params.npy", allow_pickle=True).item()
config = jnp.load("results_craftax/craftax/0_config.npy", allow_pickle=True).item()

# Set up environment
env = CraftaxSymbolicEnv()
env_params = env.default_params

# Create model
network = ActorCriticTransformer(
    action_dim=env.action_space(env_params).n,
    activation=config["ACTIVATION"],
    encoder_size=config["EMBED_SIZE"], 
    hidden_layers=config["hidden_layers"],
    num_heads=config["num_heads"],
    qkv_features=config["qkv_features"],
    num_layers=config["num_layers"],
    gating=config["gating"],
    gating_bias=config["gating_bias"],
)

# Initialize
rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
obs, state = env.reset(_rng, env_params)

# Add batch dimension to observation
obs = obs[None, :]  # Add batch dimension

# Initialize memories for the transformer with batch dimension
memories = jnp.zeros((1, config["WINDOW_MEM"], config["num_layers"], config["EMBED_SIZE"]))
memories_mask = jnp.zeros((1, config["num_heads"], 1, config["WINDOW_MEM"]+1), dtype=jnp.bool_)

# Collect frames
frames = []
frames.append(np.array(render_craftax_symbolic(state)))

# Initialize previous action and reward
prev_action = None
prev_reward = None

print("Starting simulation...")
for i in range(1000):
    # Get action from model
    rng, _rng = jax.random.split(rng)
    pi, _, memories_out = network.apply(
        params, 
        memories, 
        obs, 
        prev_action, 
        prev_reward, 
        memories_mask, 
        method=network.model_forward_eval
    )
    action = pi.sample(seed=_rng)[0]  # Remove batch dimension from action
    
    # Update memories (simple roll)
    memories = jnp.roll(memories, -1, axis=1)
    memories = memories.at[:, -1].set(memories_out)
    
    # Step environment
    rng, _rng = jax.random.split(rng)
    obs, state, reward, done, info = env.step(_rng, state, action, env_params)
    
    # Store current action and reward for next step
    prev_action = jnp.array([action])  # Add batch dimension
    prev_reward = jnp.array([reward])  # Add batch dimension
    
    # Add batch dimension to new observation
    obs = obs[None, :]
    
    # Render and store frame
    frames.append(np.array(render_craftax_symbolic(state)))
    
    if i % 100 == 0:
        print(f"Completed {i} steps")

# Save as MP4
frames_np = np.array(frames, dtype=np.uint8)
imageio.mimsave('craftax_replay.mp4', frames_np, fps=30)
print(f"Saved {len(frames)} frames to craftax_replay.mp4") 