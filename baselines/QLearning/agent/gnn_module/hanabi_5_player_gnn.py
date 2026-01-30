import jax
import jax.numpy as jnp
from functools import partial
import flax.linen as nn
from flax.linen.initializers import orthogonal
from typing import Any, Callable

# -------------------------------------------------------------------
# 1) Preprocessor for 5 Players
# -------------------------------------------------------------------
class HanabiGraphPreprocessor5Players:
    """
    Preprocessor for 5-player Hanabi.
    Hand Size: 4 cards.
    Opponents: 4.
    """
    def __init__(self):
        # -----------------------------------------------------------
        # Feature Size Calculations (Estimated based on HLE standards)
        # -----------------------------------------------------------

        # Hands:
        # 4 opponents * 4 cards * 25 features = 400
        # + 4 features for 'missing card' indicators (1 per opponent)
        self.hands_size = 404

        # Board:
        # Deck(40) + Fireworks(25) + Info(3 approx) + Life(2 approx) = 70
        # (Standard HLE board section is usually consistent)
        self.board_size = 70

        # Discards:
        # 5 colors * 10 bits = 50
        self.discards_size = 50

        # Last Action:
        # 4-player was 57. 5-player usually adds 1 bit for the extra player index encoding.
        # Estimated: 58.
        self.last_action_size = 57

        # V0 Belief:
        # Own hand size is 4.
        # 4-player used 560 (16 slots * 35 features).
        # 5-player uses 700 (20 slots * 35 features)
        self.v0_belief_size = 700

        # Total Observation Size
        self.obs_size = (
            self.hands_size
            + self.board_size
            + self.discards_size
            + self.last_action_size
            + self.v0_belief_size
        )
        # Expected: 404 + 70 + 50 + 57 + 700 = 1281

        # Note: If your specific env wrapper produces a slightly different
        # last_action size, adjust self.last_action_size accordingly.
        assert self.obs_size == 1281, f"Expected 1281, got {self.obs_size}"

        # Offsets
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
    def preprocess_observation(self, observation: jnp.ndarray) -> jnp.ndarray:
        """
        Convert a single 5-player Hanabi observation (shape: (1142,))
        into a stack of node feature vectors.
        """

        # 1) Extract Segments
        hands_feat = observation[self.hands_start:self.hands_end]        # 404
        board_feat = observation[self.board_start:self.board_end]        # 70
        discards_feat = observation[self.discards_start:self.discards_end] # 50
        last_action_feat = observation[self.last_action_start:self.last_action_end] # 58
        v0_belief_feat = observation[self.v0_belief_start:self.v0_belief_end] # 560

        # 2) Decompose Hands (4 Opponents)
        # 400 bits for cards, 4 bits for 'missing' flags
        other_hands = hands_feat[:400].reshape(4, 4, 25) # (4 players, 4 cards, 25 feats)
        hands_missing = hands_feat[400:]                 # (4,)

        # 3) Decompose Board
        deck = board_feat[:40]
        fireworks = board_feat[40:65]
        info_tokens = board_feat[65:68]
        life_tokens = board_feat[68:70]

        # 4) Decompose Discards
        discards = discards_feat.reshape(5, 10) # 5 nodes

        # 5) Decompose Last Action (heuristic split)
        # We split this roughly into semantic chunks for nodes
        la_actor = last_action_feat[:5]       # e.g. Relative actor index
        la_type = last_action_feat[5:9]       # Move type
        la_target = last_action_feat[9:14]    # Target player
        la_rest = last_action_feat[14:]       # Remaining info (color, rank, etc)

        # 6) Decompose V0 Belief
        # 560 bits / 35 features = 16 card slots (buffer)
        v0_belief_cards = v0_belief_feat.reshape(20, 35)

        # 7) Build Node List
        node_list = []

        # -- Hand Nodes --
        # 4 opponents * 4 cards = 16 nodes
        for p in range(4):
            for c in range(4):
                node_list.append(other_hands[p, c, :])

        # Add the 'missing card' flags as 1 node (or split them if preferred)
        node_list.append(hands_missing)

        # -- Board Nodes --
        node_list.append(deck)
        node_list.append(fireworks)
        node_list.append(info_tokens)
        node_list.append(life_tokens)

        # -- Discard Nodes --
        for i in range(5):
            node_list.append(discards[i])

        # -- Last Action Nodes --
        node_list.append(la_actor)
        node_list.append(la_type)
        node_list.append(la_target)
        node_list.append(la_rest)

        # -- V0 Belief Nodes --
        for i in range(20):
            node_list.append(v0_belief_cards[i])

        # 8) Pad to Max Dimension
        max_dim = max(n.shape[0] for n in node_list)
        padded_nodes = []
        for node in node_list:
            pad_len = max_dim - node.shape[0]
            padded = jnp.pad(node, (0, pad_len), constant_values=0)
            padded_nodes.append(padded)

        final_nodes = jnp.stack(padded_nodes, axis=0)
        return final_nodes

# -------------------------------------------------------------------
# 2) Observation Encoder (Generic)
# -------------------------------------------------------------------
class ObservationEncoder(nn.Module):
    num_nodes: int
    num_layers: int = 1
    obs_enc_hidden_dim: int = 64

    @nn.compact
    def __call__(self, observations):
        x = observations
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.obs_enc_hidden_dim)(x)
            x = nn.relu(x)

        # Produce adjacency logits
        logits = nn.Dense(features=self.num_nodes * self.num_nodes * 2, kernel_init=orthogonal(1.0))(x)
        logits = logits.reshape(observations.shape[0], self.num_nodes, self.num_nodes, 2)
        return logits

# -------------------------------------------------------------------
# 3) Gumbel / GCN / GraphMean (Shared/Generic)
# -------------------------------------------------------------------
class GumbelSoftmaxAdjMatrixModel(nn.Module):
    seed: int = 42
    temperature: float = 1.0

    def gumbel_softmax_sample(self, logits, rng):
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape)))
        y = logits + gumbel_noise
        return nn.softmax(y / self.temperature)

    @nn.compact
    def __call__(self, logits):
        rng = jax.random.PRNGKey(self.seed)
        # vmap sampling over batch
        gumbel_out = jax.vmap(lambda l, r: self.gumbel_softmax_sample(l, r))(
            logits, jax.random.split(rng, logits.shape[0])
        )
        return gumbel_out[..., 1] # Probability of edge existing

class GCNLayer(nn.Module):
    node_feature_dim: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        # 1. Linear Transform
        x = nn.Dense(self.node_feature_dim, kernel_init=orthogonal(1.0))(node_feats)

        # 2. Message Passing (Batch MatMul)
        # adj: (B, N, N), x: (B, N, F) -> (B, N, F)
        x = jax.vmap(jnp.matmul)(adj_matrix, x)

        # 3. Normalize
        denom = adj_matrix.sum(axis=-1, keepdims=True) + 1e-6
        x = x / denom
        return x

class GraphMean(nn.Module):
    @nn.compact
    def __call__(self, node_feats):
        return jnp.mean(node_feats, axis=1)

# -------------------------------------------------------------------
# 4) End-to-End GCN for 5 Players
# -------------------------------------------------------------------
class End2EndGCN5Players(nn.Module):
    config: Any

    def setup(self):
        # Determine number of nodes from preprocessor logic:
        # Hands: 16 (cards) + 1 (missing) = 17
        # Board: 4
        # Discards: 5
        # Last Action: 4
        # V0 Belief: 16
        # Total Nodes = 17 + 4 + 5 + 4 + 16 = 46 nodes
        self.num_nodes = 50

        self.preprocessor = HanabiGraphPreprocessor5Players()
        self.observation_encoder = ObservationEncoder(
            num_nodes=self.num_nodes,
            obs_enc_hidden_dim=self.config["OBS_ENC_HIDDEN_DIM"]
        )
        self.gumbel_model = GumbelSoftmaxAdjMatrixModel(
            seed=self.config["SEED"],
            temperature=self.config["TEMPERATURE"]
        )
        self.gcn_layer = GCNLayer(
            node_feature_dim=self.config["NODE_FEATURE_DIM"]
        )
        self.graph_mean = GraphMean()

    def __call__(self, observations):
        """
        observations: (batch_size, 1282)
        """
        # 1. Preprocess (Vector -> Nodes)
        # Output: (batch_size, 50, padded_feature_dim)
        node_feats = jax.vmap(self.preprocessor.preprocess_observation)(observations)

        # 2. Learn Adjacency (Vector -> Logits -> Adj Matrix)
        logits = self.observation_encoder(observations)
        adj_matrix = self.gumbel_model(logits)

        # 3. Graph Convolution
        node_feats = self.gcn_layer(node_feats, adj_matrix)

        # 4. Pooling
        graph_embedding = self.graph_mean(node_feats)

        return graph_embedding