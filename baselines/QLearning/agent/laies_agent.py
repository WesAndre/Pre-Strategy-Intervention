import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from agent.iql_agent import AgentRNN, ScannedRNN

class RNDNetwork(nn.Module):
    """Predictor and Target networks for RND curiosity."""
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0))(x)
        return x

class LAIESAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float
    rnd_output_dim: int = 64

    def setup(self):
        # Standard Q-learning backbone
        # This was working because variable_axes and split_rngs were set correctly
        self.agent_rnn = nn.vmap(AgentRNN, in_axes=0, out_axes=0, 
                                 variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.hidden_dim, self.init_scale
        )
        self.q_value_mlp = nn.vmap(nn.Dense, in_axes=0, out_axes=0, 
                                   variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.action_dim, kernel_init=orthogonal(self.init_scale)
        )
        
        # --- FIX APPLIED BELOW ---
        # LAIES Curiosity Modules (RND)
        # Added variable_axes and split_rngs so Flax knows how to initialize per-agent params
        
        self.rnd_target = nn.vmap(RNDNetwork, in_axes=0, out_axes=0,
                                  variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.hidden_dim, self.rnd_output_dim
        )
        
        self.rnd_predictor = nn.vmap(RNDNetwork, in_axes=0, out_axes=0, 
                                     variable_axes={"params": 0}, split_rngs={"params": 0})(
            self.hidden_dim, self.rnd_output_dim
        )

    def __call__(self, agent_hidden, obs, dones, _):
        # Q-Value calculation
        agent_hidden, agent_embedding = self.agent_rnn(agent_hidden, obs, dones)
        q_vals = self.q_value_mlp(agent_embedding)
        
        # Curiosity feature extraction
        target_feat = self.rnd_target(obs)
        pred_feat = self.rnd_predictor(obs)
        
        return agent_hidden, q_vals, (target_feat, pred_feat)