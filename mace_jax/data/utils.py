import logging
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from ase import Atoms


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


def compute_average_E0s(
    collections_train: list[Atoms], z_table: list[int]
) -> Dict[int, float]:
    """
    Function to compute the average interaction energy of each chemical element
    returns dictionary of E0s
    """
    len_train = len(collections_train)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = collections_train[i].get_potential_energy()
        for j, z in enumerate(z_table):
            A[i, j] = np.count_nonzero(collections_train[i].get_atomic_numbers() == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.error(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict
