import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Callable

class GCNLayer(nn.Module):
    node_feature_dim: int  # Output feature size per node

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        # Compute the number of neighbors for normalization
        num_neighbours = adj_matrix.sum(axis=-1, keepdims=True)
        # Apply a Dense layer to transform node features
        node_feats = nn.Dense(features=self.node_feature_dim)(node_feats)
        # Message passing: aggregate information from neighbors
        node_feats = jax.lax.batch_matmul(adj_matrix, node_feats)
        # Normalize by the number of neighbors
        node_feats = node_feats / num_neighbours
        return node_feats


class GraphMean(nn.Module):
    @nn.compact
    def __call__(self, node_feats):
        # Mean pooling over nodes to get graph-level embedding
        return jnp.mean(node_feats, axis=-2)


class End2EndGCN(nn.Module):
    node_feature_dim: int   # Feature dimension per node
    num_agents: int = 3     # Number of agents (NEW)
    num_landmarks: int = 3  # Number of landmarks (NEW)

    # # Default adjacency matrix for MPE using a default factory
    # adj_matrix_factory: Callable[[], jnp.ndarray] = lambda: jnp.array([
    #     [1, 1, 0, 0, 0, 0, 0],  # Velocity only connects to Position
    #     [1, 1, 1, 1, 1, 1, 1],  # Position connects to all other positions (landmarks, other agents)
    #     [0, 1, 1, 0, 0, 0, 0],  # Landmark 1 connected to Position
    #     [0, 1, 0, 1, 0, 0, 0],  # Landmark 2 connected to Position
    #     [0, 1, 0, 0, 1, 0, 0],  # Landmark 3 connected to Position
    #     [0, 1, 0, 0, 0, 1, 0],  # Other Agent 1 connected to Position
    #     [0, 1, 0, 0, 0, 0, 1]   # Other Agent 2 connected to Position
    # ])

    def create_adjacency_matrix(self):
        "Dynamically creating an adjacency matrix based on variable number of agents and landmarks"
        num_nodes = 2 + self.num_landmarks + (self.num_agents - 1)
        adj_matrix = jnp.zeros((num_nodes, num_nodes))

        # self connections
        adj_matrix = adj_matrix.at[jnp.diag_indices(num_nodes)].set(1)

        # Velocity (node 0) -> Position (node 1)
        adj_matrix = adj_matrix.at[0, 1].set(1)

        # Position -> All other positions
        adj_matrix = adj_matrix.at[1, :].set(1)

        # each landmark / other agent connected to position (node 1)
        for i in range(2, num_nodes):
            adj_matrix = adj_matrix.at[i, 1].set(1)
        
        return adj_matrix

    @nn.compact
    def __call__(self, observations):
        # observations shape: (time_step, batch_size, observation_size)
        # adj_matrix shape: (num_nodes, num_nodes)

        #REMOVED (harcoded to 3)
        # observations = observations[..., :14]
        # time_step, batch_size, observation_size = observations.shape
        # assert observation_size == 14, "Observation size should be 14 to split into 7 nodes of 2 features each."

        # Reshape observations to (time_step * batch_size, 7, 2)
        #node_feats = observations.reshape(-1, 7, 2)

        # observations shape: (num_agents, time_step, batch_size, gnn_obs_dim)

        num_nodes = 2 + self.num_landmarks + (self.num_agents - 1)
        correct_size = 2 * num_nodes # each node had poistion & velocity

        observations = observations[..., :correct_size]
        time_step, batch_size, observation_size = observations.shape
        assert observation_size == correct_size, f"Observation size should be {correct_size}, got {observation_size}."

        # Reshaping to (batch_size, num_nodes, 2)
        node_feats = observations.reshape(-1, num_nodes, 2)

        # Get adjacency matrix from factory
        #adj_matrix = self.adj_matrix_factory()

        adj_matrix = self.create_adjacency_matrix()
        # Tile adj_matrix to match the number of node_feats if necessary
        adj_matrix = jnp.tile(adj_matrix[None, :, :], (node_feats.shape[0], 1, 1))

        # Apply the GCN layer (one message passing step)
        node_feats = GCNLayer(node_feature_dim=self.node_feature_dim)(node_feats, adj_matrix)

        # Perform the graph readout (e.g., mean pooling)
        graph_embedding = GraphMean()(node_feats)

        # Reshape back to (time_step, batch_size, -1)
        graph_embedding = graph_embedding.reshape(time_step, batch_size, -1)
        return graph_embedding
