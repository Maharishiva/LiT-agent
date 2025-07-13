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
import wandb


# Custom TrainState to track step count
class CustomTrainState(TrainState):
    step_count: int
    env_step_count: int


from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    ThinkingWrapper,
)


from transformerXL import Transformer

class ActorCriticTransformer(nn.Module):
    # action_dim: int
    action_dim_env: int
    thinking_vocab: int
    activation: str
    hidden_layers:int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    gating:bool=False
    gating_bias:float=0.
    
    
    def setup(self):
        self.action_dim = self.action_dim_env + self.thinking_vocab
        
        # USE SETUP AND DIFFERENT FUNCTIONS BECAUSE THE TRAIN IS DIFFERENT FROM EVAL ( as we query just one step in train and don't cache memory in eval)
        
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
                                env_action_dim=self.action_dim_env,
                                thinking_vocab=self.thinking_vocab)
        
        self.actor_ln1=nn.Dense(self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_ln2= nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.actor_out= nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )
        
        
        self.critic_ln1=nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_ln2=nn.Dense(
            self.hidden_layers, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_out=nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))
        
        
        
        

    def __call__(self, memories, obs, prev_action=None, prev_reward=None, mask=None):
        x, memory_out = self.transformer(memories, obs, mask, prev_action, prev_reward)
    
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
        critic = self.critic_out(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), memory_out
    
    def model_forward_eval(self, memories, obs, prev_action=None, prev_reward=None, mask=None):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        x, memory_out = self.transformer.forward_eval(memories, obs, mask, prev_action, prev_reward)

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
        critic = self.critic_out(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), memory_out
    
    def model_forward_train(self, memories, obs, prev_action=None, prev_reward=None, mask=None): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        x = self.transformer.forward_train(memories, obs, mask, prev_action, prev_reward)

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
        critic = self.critic_out(
            critic
        )
        return pi, jnp.squeeze(critic, axis=-1)
    


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    thinking_length: jnp.ndarray
    memories_mask: jnp.ndarray
    memories_indices: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    prev_action: jnp.ndarray = None
    prev_reward: jnp.ndarray = None


    
indices_select=lambda x,y:x[y]
batch_indices_select=jax.vmap(indices_select)
roll_vmap=jax.vmap(jnp.roll,in_axes=(-2,0,None),out_axes=-2)
batchify=lambda x: jnp.reshape(x,(x.shape[0]*x.shape[1],)+x.shape[2:])


    
def make_train(config):
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    if(config["ENV_NAME"]=="craftax"):
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnvNoAutoReset
        env=CraftaxSymbolicEnvNoAutoReset()
        env_params=env.default_params
        action_dim_env = env.action_space(env_params).n
        env = LogWrapper(env)
        env = ThinkingWrapper(env, action_dim_env, config["R_THINK"])
        env = OptimisticResetVecEnvWrapper(
                env,
                num_envs=config["NUM_ENVS"],
                reset_ratio=min(16, config["NUM_ENVS"]),
            )
    else:
        env, env_params = gymnax.make(config["ENV_NAME"])
        action_dim_env = env.action_space(env_params).n 
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)
        env = ThinkingWrapper(env, action_dim_env, config["R_THINK"])
        env = BatchEnvWrapper(env,config["NUM_ENVS"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / (config["NUM_UPDATES"]) 
        return config["LR"] * frac

    
    def train(rng):

        # INIT NETWORK
        network=ActorCriticTransformer(action_dim_env=env.action_space(env_params).n,
                             thinking_vocab=config["THINKING_VOCAB"],
                             activation=config["ACTIVATION"],
                            encoder_size=config["EMBED_SIZE"],
                            hidden_layers=config["hidden_layers"],
                            num_heads=config["num_heads"],
                            qkv_features=config["qkv_features"],
                            num_layers=config["num_layers"],
                            gating=config["gating"],
                            gating_bias=config["gating_bias"],)
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((2,env.observation_space(env_params).shape[0]))
        init_memory = jnp.zeros((2,config["WINDOW_MEM"],config["num_layers"],config["EMBED_SIZE"]))
        init_mask = jnp.zeros((2,config["num_heads"],1,config["WINDOW_MEM"]+1),dtype=jnp.bool_)
        init_prev_action = jnp.zeros((2,), dtype=jnp.int32)  # Initial previous actions (zeros)
        init_prev_reward = jnp.zeros((2,))  # Initial previous rewards (zeros)
        
        network_params = network.init(_rng, init_memory, init_obs, init_prev_action, init_prev_reward, init_mask)
        
        
        
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            
            
        # Use CustomTrainState instead of TrainState to track step_count
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            step_count=0,
            env_step_count=0,
        )
        
        

        # Reset ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, None)
        #reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        #obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, memories, memories_mask, memories_mask_idx, last_obs, done, step_env_currentloop, rng, last_action, last_reward = runner_state
                
                # reset memories mask and mask idx in cask of done otherwise mask will consider one more stepif not filled (if filled= 
                memories_mask_idx = jnp.where(done, config["WINDOW_MEM"], jnp.clip(memories_mask_idx-1, 0, config["WINDOW_MEM"]))
                memories_mask = jnp.where(done[:,None,None,None], jnp.zeros((config["NUM_ENVS"], config["num_heads"], 1, config["WINDOW_MEM"]+1), dtype=jnp.bool_), memories_mask)
                
                #Update memories mask with the potential additional step taken into account at this step
                memories_mask_idx_ohot = jax.nn.one_hot(memories_mask_idx, config["WINDOW_MEM"]+1)
                memories_mask_idx_ohot = memories_mask_idx_ohot[:,None,None,:].repeat(config["num_heads"], 1)
                memories_mask = jnp.logical_or(memories_mask, memories_mask_idx_ohot)
            
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Use previous action and reward passed in runner state
                # For first step in episode (after reset), these will be zeros
                
                pi, value, memories_out = network.apply(
                    train_state.params, 
                    memories, 
                    last_obs,
                    last_action,
                    last_reward,
                    memories_mask,
                    method=network.model_forward_eval
                )
                ### mask the logits to prevent thinking if the thinking length exceeds the allowed limit ###
                logits = pi.logits
                thinking_length = env_state.thinking_length
                total_actions = logits.shape[-1]
                action_indices = jnp.arange(total_actions)
                allowed_mask = jnp.logical_or(
                    thinking_length[:, None] < config["MAX_THINKING_LEN"], # thinking len < max thinking len
                    action_indices[None, :] < network.action_dim_env, # action is env action
                )
                masked_logits = jnp.where(allowed_mask, logits, -jnp.inf)
                pi = distrax.Categorical(logits=masked_logits)
                ######
                
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                # ADD THE CACHED ACTIVATIONS IN MEMORIES FOR NEXT STEP
                memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(memories_out)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, None
                )
                
                #COMPUTE THE INDICES OF THE FINAL MEMORIES THAT ARE TAKEN INTO ACCOUNT IN THIS STEP 
                # not forgetting that we will concatenate the previous WINDOW_MEM to the NUM_STEPS so that even the first step will use some cached memory.
                #previous without this is attend to 0 which are masked but with reset happening if we start the num_steps loop during good to keep memory from previous
                memory_indices = jnp.arange(0, config["WINDOW_MEM"])[None, :] + step_env_currentloop * jnp.ones((config["NUM_ENVS"], 1), dtype=jnp.int32)
                
                transition = Transition(
                    done,
                    action,
                    value,
                    reward,
                    log_prob,
                    thinking_length,
                    memories_mask.squeeze(),
                    memory_indices,
                    last_obs,
                    info,
                    last_action,
                    last_reward,
                )
                
                # Update runner state with new action and reward for next step
                runner_state = (train_state, env_state, memories, memories_mask, memories_mask_idx, obsv, done, 
                               step_env_currentloop+1, rng, action, reward)
                
                return runner_state, (transition, memories_out)
            

            
            #also copy the first memories in memories_previous before the new rollout to concatenate previous memories with new steps so that first steps of new have memories
            memories_previous=runner_state[2]
             
            #SCAN THE STEP TO GET THE TRANSITIONS AND CACHED MEMORIES
            runner_state, (traj_batch,memories_batch) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            env_steps_batch = jnp.sum(~traj_batch.info["thinking_action"])

            # CALCULATE ADVANTAGE
            train_state, env_state, memories, memories_mask, memories_mask_idx, last_obs, done, _, rng, last_action, last_reward = runner_state
            
            # Use last action and reward from runner state
            _, last_val, _ = network.apply(
                train_state.params, 
                memories,
                last_obs,
                last_action,
                last_reward,
                memories_mask,
                method=network.model_forward_eval
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward, action = (
                        transition.done,
                        transition.value,
                        transition.reward,
                        transition.action
                    )
                    is_thinking = jnp.greater_equal(action, network.action_dim_env)
                    # reward = reward + jnp.where(is_thinking, config["R_THINK"], 0.0) # already accounted for in wrapper
                    # gamma = jnp.where(is_thinking, 1.0, config["GAMMA"])
                    gamma = config["GAMMA"]
                    delta = reward + gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + gamma * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
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
            

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                
                    traj_batch,memories_batch, advantages, targets = batch_info
                    def _loss_fn(params, traj_batch,memories_batch, gae, targets):
                        
                        
                        # USE THE CACHED MEMORIES ONLY FROM THE FIRST STEP OF A WINDOW GRAD Because all other will be computed again here.
                        #construct the memory batch from memory indices
                        memories_batch=batch_indices_select(memories_batch,traj_batch.memories_indices[:,::config["WINDOW_GRAD"]]) 
                        memories_batch=batchify(memories_batch)
                        
                        
                        #CREATE THE MASK FOR WINDOW GRAD (have to take the one from the batch and roll them to match the steps it attends
                        memories_mask=traj_batch.memories_mask.reshape((-1,config["WINDOW_GRAD"],)+traj_batch.memories_mask.shape[2:])
                        memories_mask=jnp.swapaxes(memories_mask,1,2)
                        #concatenate with 0s to fill before the roll
                        memories_mask=jnp.concatenate((memories_mask,jnp.zeros(memories_mask.shape[:-1]+(config["WINDOW_GRAD"]-1,),dtype=jnp.bool_)),axis=-1)
                        #roll of different value for each step to match the right
                        memories_mask=roll_vmap(memories_mask,jnp.arange(0,config["WINDOW_GRAD"]),-1)

                        #RESHAPE
                        obs=traj_batch.obs
                        obs=obs.reshape((-1,config["WINDOW_GRAD"] ,)+obs.shape[2:])

                        traj_batch,targets,gae=jax.tree_util.tree_map(lambda x : jnp.reshape(x,(-1,config["WINDOW_GRAD"])+x.shape[2:]),(traj_batch,targets,gae))
                      
  
                        
                        
                        # NETWORK OUTPUT
                        prev_action = traj_batch.prev_action
                        prev_reward = traj_batch.prev_reward
                        
                        pi, value = network.apply(
                            params,
                            memories_batch,
                            obs,
                            prev_action,
                            prev_reward,
                            memories_mask,
                            method=network.model_forward_train,
                        )

                        logits = pi.logits
                        t_len = traj_batch.thinking_length
                        total_actions = logits.shape[-1]
                        action_indices = jnp.arange(total_actions)
                        allowed_mask = jnp.logical_or(
                            t_len[..., None] < config["MAX_THINKING_LEN"],
                            action_indices < network.action_dim_env,
                        )
                        masked_logits = jnp.where(allowed_mask, logits, -jnp.inf)
                        pi = distrax.Categorical(logits=masked_logits)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch,memories_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch,memories_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)            
                #batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                     config["NUM_STEPS"] % config["WINDOW_GRAD"]==0
                ), "NUM_STEPS should be divi by WINDOW_GRAD to properly batch the window_grad"
                
                
                # PERMUTE ALONG THE NUM_ENVS ONLY NOT TO LOOSE TRACK FROM TEMPORAL 
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (traj_batch,memories_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x:  jnp.swapaxes(x,0,1),
                    batch,
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                
                #either create memory batch here but might be big  or send all the memeory to loss and do the things with the index in the loss
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
            
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch,memories_batch, advantages, targets, rng)
                return update_state, total_loss
            
            
            #ADD PREVIOUS WINDOW_MEM To the current NUM_STEPS SO THAT FIRST STEPS USE MEMORIES FROM PREVIOUS
            # might be a better place to add the previous memory to the traj batch to make it faster ??? 
            #or another solution is to not add it but in training means that the first element might not look at info
            memories_batch=jnp.concatenate([jnp.swapaxes(memories_previous,0,1),memories_batch],axis=0)

            
            #CRAFTAX ONLY
            metric = jax.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            metric=jax.tree_map(lambda x: x.mean(),metric)

            env_steps_total = train_state.env_step_count + env_steps_batch
            
            # Simplified wandb logging to match DQN pattern
            # Calculate timesteps - based on update count
            timesteps = runner_state[0].step_count * config["NUM_ENVS"] * config["NUM_STEPS"]
            
            # Create simple metrics dictionary with numeric values
            metrics = {
                "timesteps": timesteps,
                "update": runner_state[0].step_count,
                "return": metric["returned_episode_returns"],
                "episode_length": metric["returned_episode_lengths"],
                "thinking_count": metric["thinking_count"],
                "env_timesteps": env_steps_total,
            }
            
            # Run the update epochs
            update_state = (train_state, traj_batch,memories_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            train_state = update_state[0]

            # Increment step count and update env step count
            train_state = train_state.replace(
                step_count=train_state.step_count + 1,
                env_step_count=env_steps_total,
            )
            
            # Log to wandb if enabled
            if config.get("WANDB_MODE", "disabled") == "online":
                def callback(update, return_val, episode_length, timesteps, thinking_count, env_timesteps):
                    # Log every WANDB_LOG_FREQ updates
                    if update % config["WANDB_LOG_FREQ"] == 0:
                        wandb.log({
                            "return": float(return_val),
                            "episode_length": float(episode_length),
                            "timesteps": int(timesteps),  # This is agent steps
                            "env_timesteps": int(env_timesteps), # This is environment steps
                            "update": int(update),
                            "thinking_count": float(thinking_count),
                        })
                
                jax.debug.callback(callback, metrics["update"], metrics["return"], 
                                  metrics["episode_length"], metrics["timesteps"], metrics["thinking_count"], metrics["env_timesteps"])
            
            rng = update_state[-1]
            # Reset step_env_currentloop to 0, but keep last_action and last_reward for the next batch
            runner_state = (train_state, env_state, memories, memories_mask, memories_mask_idx, 
                           last_obs, done, 0, rng, last_action, last_reward)
            return runner_state, metric
        
        
        # INITIALIZE the memories and memories mask 
        rng, _rng = jax.random.split(rng)
        memories = jnp.zeros((config["NUM_ENVS"], config["WINDOW_MEM"], config["num_layers"], config["EMBED_SIZE"]))
        memories_mask = jnp.zeros((config["NUM_ENVS"], config["num_heads"], 1, config["WINDOW_MEM"]+1), dtype=jnp.bool_)
        # memories +1 bc will remove one 
        memories_mask_idx = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32) + (config["WINDOW_MEM"]+1)
        done = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.bool_)
        
        # Initialize previous actions and rewards with zeros
        init_action = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
        init_reward = jnp.zeros((config["NUM_ENVS"],))
        
        runner_state = (train_state, env_state, memories, memories_mask, memories_mask_idx, 
                        obsv, done, 0, _rng, init_action, init_reward)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train