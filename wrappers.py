import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any

#Taken from https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/wrappers.py

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class ThinkingEnvState:
    env_state: Any
    thinking_count: int
    thinking_length: int
    last_obs: Any
    last_info: Any

class ThinkingWrapper(GymnaxWrapper):
    def __init__(self, env, env_action_dim: int, think_reward: float = -0.005):
        super().__init__(env)
        self.env_action_dim = int(env_action_dim)
        self.think_reward = float(think_reward)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, inner_state = self._env.reset(key, params)

        info_template = {
            "returned_episode_returns": jnp.array(0.0,  jnp.float32),
            "returned_episode_lengths": jnp.array(0,    jnp.int32),
            "timestep":                jnp.array(0,    jnp.int32),
            "returned_episode":        jnp.array(False),
            "thinking_action":         jnp.array(False),
            "thinking_count":          jnp.array(0,    jnp.int32),
        }
        state = ThinkingEnvState(inner_state, 0, 0, obs, info_template)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state: ThinkingEnvState, action, params=None):
        is_think = action >= self.env_action_dim
        operand = (key, state, action, params)
        def _think(op):
            key, st, act, prm = op
            info = dict(st.last_info)
            total_count = st.thinking_count + 1
            thinking_length = st.thinking_length + 1
            info["thinking_action"] = True
            info["thinking_count"]  = total_count
            reward = jnp.asarray(self.think_reward, jnp.float32)
            done   = jnp.array(False)
            new_state = ThinkingEnvState(
                st.env_state, total_count, thinking_length, st.last_obs, info
            )
            return st.last_obs, new_state, reward, done, info


        def _act(op):
            key, st, act, prm = op
            obs, inner_state, reward, done, info_inner = self._env.step(
                key, st.env_state, act, prm
            )

            info = dict(st.last_info)
            info["returned_episode_returns"]  = info_inner["returned_episode_returns"]
            info["returned_episode_lengths"]  = info_inner["returned_episode_lengths"]
            info["returned_episode"]          = info_inner["returned_episode"]
            info["timestep"]                  = info_inner["timestep"]
            info["thinking_action"]           = False
            info["thinking_count"]            = st.thinking_count

            new_state = ThinkingEnvState(
                inner_state, st.thinking_count, 0, obs, info
            )
            return obs, new_state, reward, done, info

        return jax.lax.cond(is_think, _think, _act, operand)
    
@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info

