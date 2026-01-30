import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from typing import Sequence, Dict

# 1. The Standard Agent Implementation (Same as IQL but wrapped here to avoid imports)
class ScannedRNN(nn.Module):
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )

class AgentRNN(nn.Module):
    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, hidden, x, resets):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        x = nn.relu(x)
        rnn_state, embedding = ScannedRNN()(hidden, (x, resets))
        return rnn_state, embedding

# 2. The Intrinsic Reward Proxy Network
class IntrinsicProxy(nn.Module):
    hidden_dim: int
    
    @nn.compact
    def __call__(self, obs, action_one_hot):
        # The proxy takes observation and the chosen action to decide a reward
        x = jnp.concatenate([obs, action_one_hot], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        # Output a single scalar reward
        r_in = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return r_in

# 3. The Combined LIIR Agent Class
class LIIRAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float

    def setup(self):
        # Policy Network (Agent)
        self.agent_rnn = nn.vmap(AgentRNN, in_axes=0, out_axes=0, 
                                 variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.hidden_dim, self.init_scale
        )
        self.q_value_mlp = nn.vmap(nn.Dense, in_axes=0, out_axes=0, 
                                   variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.action_dim, kernel_init=orthogonal(self.init_scale)
        )
        
        # Proxy Network (Meta-Learner)
        # Shared across agents or individual? Usually individual for heterogeneity.
        self.proxy = nn.vmap(IntrinsicProxy, in_axes=0, out_axes=0,
                             variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.hidden_dim
        )

    def __call__(self, hidden, obs, dones, actions_one_hot=None):
        # Forward pass for the Agent (Get Q-values)
        new_hidden, embedding = self.agent_rnn(hidden, obs, dones)
        q_vals = self.q_value_mlp(embedding)
        
        # Forward pass for the Proxy (Get Intrinsic Rewards)
        # Only compute if actions are provided (during updates)
        r_in = None
        if actions_one_hot is not None:
            r_in = self.proxy(obs, actions_one_hot)
            
        return new_hidden, q_vals, r_in