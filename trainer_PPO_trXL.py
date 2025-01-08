import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from gymnax.wrappers.purerl import  FlattenObservationWrapper

from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
)

from transformerXL import Transformer

# ===== NEW IMPORTS for wandb & callbacks =====
import wandb
from jax.experimental import io_callback


class ActorCriticTransformer(nn.Module):
    action_dim: Sequence[int]
    activation: str
    hidden_layers: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    gating: bool = False
    gating_bias: float = 0.0

    def setup(self):
        # USE SETUP AND DIFFERENT FUNCTIONS BECAUSE THE TRAIN IS DIFFERENT FROM EVAL
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias,
        )

        self.actor_ln1 = nn.Dense(
            self.hidden_layers,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_ln2 = nn.Dense(
            self.hidden_layers,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.actor_out = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

        self.critic_ln1 = nn.Dense(
            self.hidden_layers,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_ln2 = nn.Dense(
            self.hidden_layers,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
        self.critic_out = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, memories, obs, mask):
        x, memory_out = self.transformer(memories, obs, mask)

        actor_mean = self.actor_ln1(x)
        actor_mean = self.activation_fn(actor_mean)
        actor_mean = self.actor_ln2(actor_mean)
        actor_mean = self.activation_fn(actor_mean)
        actor_mean = self.actor_out(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = self.critic_ln1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_ln2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1), memory_out

    def model_forward_eval(self, memories, obs, mask):
        """Used during environment rollout (single timestep of obs)."""
        x, memory_out = self.transformer.forward_eval(memories, obs, mask)

        actor_mean = self.actor_ln1(x)
        actor_mean = self.activation_fn(actor_mean)
        actor_mean = self.actor_ln2(actor_mean)
        actor_mean = self.activation_fn(actor_mean)
        actor_mean = self.actor_out(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = self.critic_ln1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_ln2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)

        return pi, jnp.squeeze(critic, axis=-1), memory_out

    def model_forward_train(self, memories, obs, mask):
        """Used during training with a window of obs; no memory returned."""
        x = self.transformer.forward_train(memories, obs, mask)

        actor_mean = self.actor_ln1(x)
        actor_mean = self.activation_fn(actor_mean)
        actor_mean = self.actor_ln2(actor_mean)
        actor_mean = self.activation_fn(actor_mean)
        actor_mean = self.actor_out(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = self.critic_ln1(x)
        critic = self.activation_fn(critic)
        critic = self.critic_ln2(critic)
        critic = self.activation_fn(critic)
        critic = self.critic_out(critic)
        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


indices_select = lambda x, y: x[y]
batch_indices_select = jax.vmap(indices_select)
roll_vmap = jax.vmap(jnp.roll, in_axes=(-2, 0, None), out_axes=-2)
batchify = lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:])


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if config["ENV_NAME"] == "craftax":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset

        env = CraftaxSymbolicEnvNoAutoReset()
        env_params = env.default_params
        env = LogWrapper(env)
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(16, config["NUM_ENVS"]),
        )
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)
        env = BatchEnvWrapper(env, config["NUM_ENVS"])

    def linear_schedule(count):
        frac = 1.0 - (
            (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / (config["NUM_UPDATES"])
        )
        return config["LR"] * frac

    def train(rng):

        # ====== Initialize wandb ======
        wandb.init(
            project="LiT-agent-experimental",
            config=config,
            name="run-" + str(config["seed"])  
        )

        # INIT NETWORK
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
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((2, env.observation_space(env_params).shape[0]))
        init_memory = jnp.zeros(
            (2, config["WINDOW_MEM"], config["num_layers"], config["EMBED_SIZE"])
        )
        init_mask = jnp.zeros(
            (2, config["num_heads"], 1, config["WINDOW_MEM"] + 1), dtype=jnp.bool_
        )
        network_params = network.init(_rng, init_memory, init_obs, init_mask)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Reset ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    memories,
                    memories_mask,
                    memories_mask_idx,
                    last_obs,
                    done,
                    step_env_currentloop,
                    rng,
                ) = runner_state

                # reset memories / mask if done
                memories_mask_idx = jnp.where(
                    done,
                    config["WINDOW_MEM"],
                    jnp.clip(memories_mask_idx - 1, 0, config["WINDOW_MEM"]),
                )
                memories_mask = jnp.where(
                    done[:, None, None, None],
                    jnp.zeros(
                        (
                            config["NUM_ENVS"],
                            config["num_heads"],
                            1,
                            config["WINDOW_MEM"] + 1,
                        ),
                        dtype=jnp.bool_,
                    ),
                    memories_mask,
                )

                # update memories_mask
                memories_mask_idx_ohot = jax.nn.one_hot(
                    memories_mask_idx, config["WINDOW_MEM"] + 1
                )
                memories_mask_idx_ohot = memories_mask_idx_ohot[:, None, None, :].repeat(
                    config["num_heads"], 1
                )
                memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value, memories_out = network.apply(
                    train_state.params,
                    memories,
                    last_obs,
                    memories_mask,
                    method=network.model_forward_eval,
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # ADD the new memory
                memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                # memory indices
                memory_indices = jnp.arange(0, config["WINDOW_MEM"])[None, :] + (
                    step_env_currentloop * jnp.ones((config["NUM_ENVS"], 1), dtype=jnp.int32)
                )

                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    log_prob,
                    memories_mask.squeeze(),
                    memory_indices,
                    last_obs,
                    info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    memories,
                    memories_mask,
                    memories_mask_idx,
                    obsv,
                    done,
                    step_env_currentloop + 1,
                    rng,
                )
                return runner_state, (transition, memories_out)

            memories_previous = runner_state[2]
            runner_state, (traj_batch, memories_batch) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # compute advantages
            (
                train_state,
                env_state,
                memories,
                memories_mask,
                memories_mask_idx,
                last_obs,
                done,
                _,
                rng,
            ) = runner_state
            _, last_val, _ = network.apply(
                train_state.params,
                memories,
                last_obs,
                memories_mask,
                method=network.model_forward_eval,
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # update epoch
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, memories_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, memories_batch, gae, targets):
                        # retrieve memories for each step
                        mem_b = batch_indices_select(
                            memories_batch, traj_batch.memories_indices[:, :: config["WINDOW_GRAD"]]
                        )
                        mem_b = batchify(mem_b)

                        # create the mask for window grad
                        memories_mask_local = traj_batch.memories_mask.reshape(
                            (-1, config["WINDOW_GRAD"])
                            + traj_batch.memories_mask.shape[2:]
                        )
                        memories_mask_local = jnp.swapaxes(memories_mask_local, 1, 2)
                        memories_mask_local = jnp.concatenate(
                            (
                                memories_mask_local,
                                jnp.zeros(
                                    memories_mask_local.shape[:-1]
                                    + (config["WINDOW_GRAD"] - 1,),
                                    dtype=jnp.bool_,
                                ),
                            ),
                            axis=-1,
                        )
                        memories_mask_local = roll_vmap(
                            memories_mask_local, jnp.arange(0, config["WINDOW_GRAD"]), -1
                        )

                        # reshape obs
                        obs = traj_batch.obs
                        obs = obs.reshape((-1, config["WINDOW_GRAD"]) + obs.shape[2:])

                        traj_batch_res, targets_res, gae_res = jax.tree_util.tree_map(
                            lambda x: jnp.reshape(
                                x, (-1, config["WINDOW_GRAD"]) + x.shape[2:]
                            ),
                            (traj_batch, targets, gae),
                        )

                        pi, value = network.apply(
                            params,
                            mem_b,
                            obs,
                            memories_mask_local,
                            method=network.model_forward_train,
                        )
                        log_prob = pi.log_prob(traj_batch_res.action)

                        # value loss
                        value_pred_clipped = traj_batch_res.value + (
                            value - traj_batch_res.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets_res)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_res)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # actor loss
                        ratio = jnp.exp(log_prob - traj_batch_res.log_prob)
                        gae_norm = (gae_res - gae_res.mean()) / (gae_res.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = (
                            jnp.clip(
                                ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                            )
                            * gae_norm
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss_val, aux_losses), grads = grad_fn(
                        train_state.params,
                        traj_batch,
                        memories_batch,
                        advantages,
                        targets,
                    )
                    value_loss, policy_loss, entropy = aux_losses
                    train_state = train_state.apply_gradients(grads=grads)

                    # Return full set for logging
                    return train_state, (loss_val, value_loss, policy_loss, entropy)

                (
                    train_state,
                    traj_batch,
                    memories_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                assert (
                    config["NUM_STEPS"] % config["WINDOW_GRAD"] == 0
                ), "NUM_STEPS should be divisible by WINDOW_GRAD."

                # PERMUTE ALONG NUM_ENVS
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (traj_batch, memories_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), batch)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                # Scan over minibatches => (total_loss, value_loss, policy_loss, entropy)
                train_state, losses_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                # losses_info shape => (NUM_MINIBATCHES, 4)

                # Average over minibatches
                total_loss_arr, val_loss_arr, pol_loss_arr, ent_arr = losses_info.T
                avg_total_loss = jnp.mean(total_loss_arr)
                avg_val_loss = jnp.mean(val_loss_arr)
                avg_pol_loss = jnp.mean(pol_loss_arr)
                avg_entropy = jnp.mean(ent_arr)

                # current LR if annealing
                current_lr = (
                    linear_schedule(train_state.step) if config["ANNEAL_LR"] else config["LR"]
                )

                # We can log "episodic_return" from the metric below
                # (We will define the callback at the end)

                # Return updated state and log details
                update_state = (
                    train_state,
                    traj_batch,
                    memories_batch,
                    advantages,
                    targets,
                    rng,
                )
                # Return the averaged losses from this epoch
                return update_state, (
                    avg_total_loss,
                    avg_val_loss,
                    avg_pol_loss,
                    avg_entropy,
                    current_lr,
                )

            # Concatenate old memory with new memory
            memories_batch = jnp.concatenate([jnp.swapaxes(memories_previous, 0, 1), memories_batch], axis=0)

            # metric for craftax only
            metric = jax.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            metric = jax.tree_map(lambda x: x.mean(), metric)

            update_state = (train_state, traj_batch, memories_batch, advantages, targets, rng)
            update_state, losses_info_epoch = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # losses_info_epoch => shape (UPDATE_EPOCHS, 5) (since we returned 5 items)
            # parse final set of stats from last epoch
            avg_total_loss, avg_val_loss, avg_pol_loss, avg_entropy, current_lr = losses_info_epoch[-1]

            # Now do wandb logging via io_callback
            # We also log `returned_episode_returns` from 'metric'.
            # Pack data into a single device array
            log_data = jnp.array(
                [
                    avg_val_loss,
                    avg_pol_loss,
                    avg_entropy,
                    avg_total_loss,
                    metric["returned_episode_returns"],
                    current_lr,
                    train_state.step.astype(jnp.float32),
                ],
                dtype=jnp.float32,
            )

            def _wandb_logging_fn(host_arr):
                arr = np.array(host_arr, dtype=np.float32)
                wandb.log(
                    {
                        "value_loss": float(arr[0]),
                        "policy_loss": float(arr[1]),
                        "entropy": float(arr[2]),
                        "total_loss": float(arr[3]),
                        "episodic_return": float(arr[4]),
                        "learning_rate": float(arr[5]),
                        "update_step": int(arr[6]),
                    }
                )

            _ = io_callback(
                _wandb_logging_fn,
                log_data,
                result_shape=(),
                ordered=True,  # keep ordering
            )

            # Return the new runner_state, plus the "metric" for plotting
            runner_state = (
                train_state,
                env_state,
                memories,
                memories_mask,
                memories_mask_idx,
                last_obs,
                done,
                0,
                rng,
            )
            return runner_state, metric

        # Initialize memory
        rng, _rng = jax.random.split(rng)
        memories = jnp.zeros(
            (config["NUM_ENVS"], config["WINDOW_MEM"], config["num_layers"], config["EMBED_SIZE"])
        )
        memories_mask = jnp.zeros(
            (config["NUM_ENVS"], config["num_heads"], 1, config["WINDOW_MEM"] + 1), dtype=jnp.bool_
        )
        memories_mask_idx = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32) + (
            config["WINDOW_MEM"] + 1
        )
        done = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.bool_)

        runner_state = (
            train_state,
            env_state,
            memories,
            memories_mask,
            memories_mask_idx,
            obsv,
            done,
            0,
            _rng,
        )
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train
