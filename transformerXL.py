import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rel_multi_head import RelMultiHeadDotProductAttention

# CODE IS HEAVILY INSPIRED FROM https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/deprecated/transfo_xl/modeling_transfo_xl.py
# MOST OF THE TIME JUST A CONVERSION IN JAX
# AS WELL AS INSPIRATIONS FROM https://github.com/MarcoMeter/episodic-transformer-memory-ppo

class Gating(nn.Module):
    #code taken from https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    d_input:int
    bg:float=0.
    @nn.compact
    def __call__(self,x,y):
        r = jax.nn.sigmoid(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(x))
        z = jax.nn.sigmoid(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(x) - self.param('gating_bias',constant(self.bg),(self.d_input,)))
        h = jnp.tanh(nn.Dense(self.d_input,use_bias=False)(y) + nn.Dense(self.d_input,use_bias=False)(r*x))
        g = (1 - z)* x + (z*  h)
        return g


class transformer_layer(nn.Module):
    num_heads: int
    out_features: int
    qkv_features: int
    gating:bool =False
    gating_bias:float =0.

    def setup(self):
        self.attention1 = RelMultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.qkv_features,
                                           out_features=self.out_features)

        self.ln1 = nn.LayerNorm()

        self.dense1 = nn.Dense(self.out_features)

        self.dense2 = nn.Dense(self.out_features)

        self.ln2 = nn.LayerNorm()
        if(self.gating):
            self.gate1=Gating(self.out_features,self.gating_bias)
            self.gate2=Gating(self.out_features,self.gating_bias)

        
        
    def __call__(self, values_keys:jnp.ndarray, queries:jnp.ndarray, pos_embed:jnp.ndarray, mask: jnp.ndarray):
        
        
        ### Post norm
        
        #out_attention = queries+ self.attention1(inputs_kv=keys,inputs_q=queries,mask=mask)
        #out_attention = self.ln1(out_attention)

        #out = self.dense1(out_attention)
        #out = nn.activation.relu(out)
        #out = self.dense2(out_attention)

        #out = out + out_attention

        #out = self.ln2(out)
        
        #pre norm
        values_keys=self.ln1(values_keys)
        queries_n=self.ln1(queries)
        attention= self.attention1(inputs_kv=values_keys,inputs_q=queries_n,mask=mask,pos_embed=pos_embed)
        if(self.gating):
            out_attention= self.gate1(queries,jax.nn.relu(attention))
        else:
            out_attention = queries+ attention

        out_attention_n=self.ln2(out_attention)
        out = self.dense1(out_attention_n)
        out = nn.activation.gelu(out)
        #out = nn.activation.relu(out)
        out = self.dense2(out)
        if(self.gating):
            out= self.gate2(out,jax.nn.relu(out_attention))
        else:
            out = out + out_attention


        return out





    

class PositionalEmbedding(nn.Module):
    dim_emb:int
    def setup(self):

        self.inv_freq = 1 / (10000 ** (jnp.arange(0.0, self.dim_emb, 2.0) / self.dim_emb))

    def __call__(self, pos_seq, bsz=None):
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        pos_emb = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)

        #if bsz is not None:
        #    return pos_emb[:, None, :].expand(-1, bsz, -1)
        #else:
        #    return pos_emb[:, None, :]
        return pos_emb
                        
                        
class Transformer(nn.Module):
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers: int
    env_action_dim: int  # number of real environment actions
    thinking_vocab: int  # number of thinking actions
    gating: bool = False
    gating_bias: float = 0.

    @property
    def total_action_dim(self):
        return self.env_action_dim + self.thinking_vocab

    def setup(self):
        self.encoder = nn.Dense(self.encoder_size)
        
        # separate embeddings for environment and thinking actions
        self.env_action_embed = nn.Embed(num_embeddings=self.env_action_dim, features=self.encoder_size)
        self.thinking_embed = nn.Embed(num_embeddings=self.thinking_vocab, features=self.encoder_size)
        self.reward_embed = nn.Dense(self.encoder_size)
        self.token_projection = nn.Dense(self.encoder_size)
        
        self.tf_layers = [transformer_layer(num_heads=self.num_heads, qkv_features=self.qkv_features,
                                           out_features=self.encoder_size,
                                           gating=self.gating, gating_bias=self.gating_bias) for _ in range(self.num_layers)]
        
        self.pos_emb = PositionalEmbedding(self.encoder_size)

    def _create_combined_token(self, obs, prev_action, prev_reward):
        """Create a token from observation and (prev_action, prev_reward).

        If prev_action corresponds to a thinking action, the token is *only* the
        thinking-action embedding.  Otherwise the token is the projected
        concatenation of [state_emb, action_emb, reward_emb].
        """
        state_emb = self.encoder(obs)

        # Determine whether the previous action was a thinking action.
        is_think = jnp.greater_equal(prev_action, self.env_action_dim)  # bool array

        # Compute indices for embedding look-ups (keep them in-range regardless of branch)
        env_idx = jnp.where(is_think, 0, prev_action)
        think_idx = jnp.where(is_think, prev_action - self.env_action_dim, 0)

        env_action_emb = self.env_action_embed(env_idx)
        thinking_emb = self.thinking_embed(think_idx)

        action_token = jnp.where(is_think[..., None], thinking_emb, env_action_emb)

        # Embed reward (always used in non-thinking branch)
        reward_emb = self.reward_embed(prev_reward[..., None] if state_emb.ndim > prev_reward.ndim else prev_reward)

        is_think_bit = is_think.astype(jnp.float32)[..., None]

        token = self.token_projection(jnp.concatenate([state_emb, action_token, reward_emb, is_think_bit], axis=-1))
        
        return token
    
    def __call__(self, memories, obs, mask, prev_action=None, prev_reward=None):
        # Create token from obs, prev_action, and prev_reward
        x = self._create_combined_token(obs, prev_action, prev_reward)
        pos_embed = self.pos_emb(jnp.arange(1+memories.shape[-3],-1,-1))[:1+memories.shape[-3]]

        i = 0
        for layer in self.tf_layers:
            memory = jnp.concatenate([memories[:,:,i], x[:,None]], axis=-2)
            x = layer(values_keys=memory, queries=x[:,None], pos_embed=pos_embed, mask=mask)
            x = x.squeeze()
            i = i+1
            
        return x

    def forward_eval(self, memories, obs, mask, prev_action=None, prev_reward=None):
        # Create token from obs, prev_action, and prev_reward
        x = self._create_combined_token(obs, prev_action, prev_reward)
        
        out_memory = jnp.zeros((x.shape[0], self.num_layers) + x.shape[1:])
        i = 0
        
        pos_embed = self.pos_emb(jnp.arange(1+memories.shape[-3],-1,-1))[:1+memories.shape[-3]]      
        
        for layer in self.tf_layers:
            out_memory = out_memory.at[:,i].set(x)
            
            memory = jnp.concatenate([memories[:,:,i], x[:,None]], axis=-2)
            x = layer(values_keys=memory, pos_embed=pos_embed, queries=x[:,None], mask=mask)
            x = x.squeeze()
            i = i+1
            
        return x, out_memory
    
    def forward_train(self, memories, obs, mask, prev_action=None, prev_reward=None):
        # Create token from obs, prev_action, and prev_reward
        x = self._create_combined_token(obs, prev_action, prev_reward)
        
        pos_embed = self.pos_emb(jnp.arange(x.shape[-2]+memories.shape[-3],-1,-1))[:x.shape[-2]+memories.shape[-3]]
        
        i = 0
        for layer in self.tf_layers:
            memory = jnp.concatenate([jnp.take(memories, i, -2), x], axis=-2)
            x = layer(values_keys=memory, pos_embed=pos_embed, queries=x, mask=mask)
            i = i+1

        return x
    
    
    
