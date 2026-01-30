import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
from flax.linen.initializers import orthogonal
from typing import Any, Callable

class Hanabi5PlayerPreprocessor:
    def __init__(self):
        # Feature sizes for 5-player Hanabi (Standard Encoding)
        self.hands_size = 402        # (4 other players * 4 cards * 25) + 2 missing card bits
        self.board_size = 76         # Deck(40), Fireworks(25), Info(8), Life(3)
        self.discards_size = 50      # 5 colors * 10 bits
        self.last_action_size = 64   # Action/Target indices expanded for 5 agents
        self.v0_belief_size = 700    # 20 cards * 35 features
        
        # Total should be 1292. 
        # Note: If your wrapper augments this, it adds to the END of the vector.
        self.obs_size = self.hands_size + self.board_size + self.discards_size + self.last_action_size + self.v0_belief_size
        
        # Define the splits
        self.hands_start = 0
        self.hands_end = self.hands_start + self.hands_size

        self.board_start = self.hands_end
        self.board_end = self.board_start + self.board_size

        self.discards_start = self.board_end
        self.discards_end = self.discards_start + self.discards_size

        self.last_action_start = self.discards_end
        self.last_action_end = self.last_action_start + self.last_action_size

        self.v0_belief_start = self.last_action_end
        self.v0_belief_end = self.v0_belief_start + self.v0_belief_size

    @partial(jax.jit, static_argnums=(0,))
    def preprocess_observation(self, observation):
        """
        Convert a 5-player observation (shape: (1292,)) into node features.
        """
        # Extract segments
        hands_feat = observation[self.hands_start:self.hands_end]
        board_feat = observation[self.board_start:self.board_end]
        discards_feat = observation[self.discards_start:self.discards_end]
        last_action_feat = observation[self.last_action_start:self.last_action_end]
        v0_belief_feat = observation[self.v0_belief_start:self.v0_belief_end]

        # Decompose Hands: 4 players * 4 cards = 16 cards
        other_player_hands = hands_feat[:400].reshape(16, 25)  # 16 nodes
        hands_missing_card = hands_feat[400:]  # shape (2,) - 1 node

        # Decompose Board
        deck = board_feat[0:40]
        fireworks = board_feat[40:65]
        info_tokens = board_feat[65:73]
        life_tokens = board_feat[73:76]

        # Discards
        discards = discards_feat.reshape(5, 10)  # 5 nodes

        # Last Action (Indices adjusted for 5 players)
        la_acting_player = last_action_feat[0:5]    # One-hot player index
        la_movetype = last_action_feat[5:9]         # Move type
        la_target_player = last_action_feat[9:14]   # Target player index
        la_color_revealed = last_action_feat[14:19]
        la_rank_revealed = last_action_feat[19:24]
        la_reveal_outcome = last_action_feat[24:29]
        la_position = last_action_feat[29:34]
        la_card_played_discarded = last_action_feat[34:59]
        la_card_played_scored = last_action_feat[59:60]
        la_card_played_info = last_action_feat[60:61]
        # Bits 61-63 are padding

        # V0 Belief: 20 cards * 35 features
        # 500 bits (possible card) + 100 bits (color hint) + 100 bits (rank hint)
        possible_card = v0_belief_feat[:500].reshape(20, 25)
        color_hinted = v0_belief_feat[500:600].reshape(20, 5)

        rank_slice = v0_belief_feat[600:700]
        # Pad with zeros if the environment provides fewer bits than expected
        rank_slice = jnp.pad(rank_slice, (0, 100 - rank_slice.shape[0]))
        rank_hinted = rank_slice.reshape(20, 5)
        v0_belief_nodes = jnp.concatenate([possible_card, color_hinted, rank_hinted], axis=-1)

        # Build Node List
        node_list = []
        for i in range(16):
            node_list.append(other_player_hands[i])
        node_list.append(hands_missing_card)

        node_list.extend([deck, fireworks, info_tokens, life_tokens])

        for i in range(5):
            node_list.append(discards[i])

        la_nodes = [la_acting_player, la_movetype, la_target_player, la_color_revealed,
                    la_rank_revealed, la_reveal_outcome, la_position, la_card_played_discarded,
                    la_card_played_scored, la_card_played_info]
        node_list.extend(la_nodes)

        for i in range(20):
            node_list.append(v0_belief_nodes[i])

        # Pad all to max_dim (40)
        max_dim = 40
        padded_nodes = [jnp.pad(n, (0, max_dim - n.shape[0])) for n in node_list]

        return jnp.stack(padded_nodes, axis=0)

# Observation Encoder Class
class ObservationEncoder(nn.Module):
    """Encodes observations into logits for adjacency."""
    num_nodes: int
    num_layers: int = 1
    obs_enc_hidden_dim: int = 64  # increased dimension for complexity

    @nn.compact
    def __call__(self, observations):
        x = observations
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.obs_enc_hidden_dim)(x)
            x = nn.relu(x)
        logits = nn.Dense(features=self.num_nodes * self.num_nodes * 2, kernel_init=orthogonal(1.0))(x)
        logits = logits.reshape(observations.shape[0], self.num_nodes, self.num_nodes, 2)
        return logits

# Gumbel Softmax Adjacency Matrix Model
class GumbelSoftmaxAdjMatrixModel(nn.Module):
    seed: int = 1
    temperature: float = 1.0

    def gumbel_softmax_sample(self, logits, rng):
        # Sample Gumbel noise
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape)))
        y = logits + gumbel_noise
        return nn.softmax(y / self.temperature)

    @nn.compact
    def __call__(self, logits):
        # Get RNG
        rng = jax.random.PRNGKey(self.seed)
        # vmap over batch dimension
        gumbel_softmax_output = (jax.vmap(lambda l, r: self.gumbel_softmax_sample(l, r))
                                 (logits, jax.random.split(rng, logits.shape[0])))
        # Extract "edge" probability
        soft_adj_matrix = gumbel_softmax_output[..., 1]
        return soft_adj_matrix

# GCN Layer Class
class GCNLayer(nn.Module):
    node_feature_dim: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        num_neighbours = adj_matrix.sum(axis=-1, keepdims=True)
        node_feats = nn.Dense(features=self.node_feature_dim, kernel_init=orthogonal(1.0))(node_feats)
        node_feats = jax.vmap(lambda adj, feat: jnp.dot(adj, feat))(adj_matrix, node_feats)
        node_feats = node_feats / (num_neighbours + 1e-6)
        return node_feats

# Graph Mean Pooling Class
class GraphMean(nn.Module):
    @nn.compact
    def __call__(self, node_feats):
        return jnp.mean(node_feats, axis=-2)

# End-to-End GCN Module
class End2EndGCN(nn.Module):
    config: Any
    obs_enc_hidden_dim: int = 64
    temperature: float = 1.0


    def setup(self):
        # For Hanabi: Based on the preprocessor, we have 46 nodes
        self.num_nodes = 56
        self.preprocessor = Hanabi5PlayerPreprocessor()
        self.observation_encoder = ObservationEncoder(num_nodes=self.num_nodes,
                                                      obs_enc_hidden_dim=self.config["OBS_ENC_HIDDEN_DIM"],)
        self.gumbel_softmax_model = GumbelSoftmaxAdjMatrixModel(seed=self.config['SEED'],
                                                                temperature=self.config["TEMPERATURE"])
        self.gcn_layer = GCNLayer(node_feature_dim=self.config["NODE_FEATURE_DIM"],)
        self.graph_mean = GraphMean()

    def __call__(self, observations):
        """
        observations: jnp.ndarray with shape (time_step, batch_size, 658)
        rng: jax.random.PRNGKey
        """
        # observations shape: (time_step, batch_size, 658)
        time_step, batch_size, obs_size = observations.shape

        observations = observations.reshape(time_step * batch_size, obs_size)

        # Preprocess into node features
        node_feats = jax.vmap(self.preprocessor.preprocess_observation)(observations)
        # node_feats shape: (time_step * batch_size, num_nodes, max_dim)

        # Encode observations to produce adjacency logits
        logits = self.observation_encoder(observations)
        # logits shape: (time_step * batch_size, num_nodes, num_nodes, 2)

        # Generate soft adjacency matrix
        adj_matrix = self.gumbel_softmax_model(logits)
        # adj_matrix shape: (time_step * batch_size, num_nodes, num_nodes)

        # GCN Layer
        node_feats = self.gcn_layer(node_feats, adj_matrix)
        # node_feats: (time_step * batch_size, num_nodes, c_out)

        # Mean pooling
        graph_embedding = self.graph_mean(node_feats)
        graph_embedding = graph_embedding.reshape(time_step, batch_size, -1)

        return graph_embedding

