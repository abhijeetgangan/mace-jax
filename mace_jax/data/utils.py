from typing import Tuple

import jax.numpy as jnp


def get_edge_vectors_and_lengths(
    positions: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    shifts: jnp.ndarray,
    normalize: bool = False,
    eps: float = 1e-10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get edge vectors and lengths between senders and receivers.

    Parameters
    ----------
    positions : jnp.ndarray
        Node positions with shape [num_nodes, 3].
    senders : jnp.ndarray
        Indices of sender nodes with shape [num_edges].
    receivers : jnp.ndarray
        Indices of receiver nodes with shape [num_edges].
    shifts : jnp.ndarray
        Shift vectors for periodic boundary conditions with shape [num_edges, 3].
    normalize : bool, optional
        Whether to normalize edge vectors, by default False.
    eps : float, optional
        Small constant to avoid division by zero, by default 1e-10.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        Edge vectors with shape [num_edges, 3] and edge lengths with shape [num_edges].
    """
    edge_vectors = positions[receivers] - positions[senders] + shifts  # [num_edges, 3]
    edge_lengths = jnp.linalg.norm(edge_vectors, axis=-1)  # [num_edges]

    if normalize:
        edge_vectors = edge_vectors / (edge_lengths[..., None] + eps)
    return edge_vectors, edge_lengths
