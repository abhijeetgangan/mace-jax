import jax
import jax.numpy as jnp
import numpy as np
import cuequivariance as cue

from mace_jax.modules.models import MACEModel

# Dataset specifications
num_species = 5
num_graphs = 10
avg_num_neighbors = 20

model_size = "MP-S"

num_atoms = 3_000
num_edges = 160_000

model = MACEModel(
    num_layers=2,
    num_features={
        "MP-S": 128,
        "MP-M": 128,
        "MP-L": 128,
        "OFF-S": 64 + 32,
        "OFF-M": 128,
        "OFF-L": 128 + 64,
    }[model_size],
    num_species=num_species,
    max_ell=3,
    correlation=3,
    num_radial_basis=8,
    interaction_irreps=cue.Irreps(cue.O3, "0e+1o+2e+3o"),
    hidden_irreps=cue.Irreps(
        cue.O3,
        {
            "MP-S": "0e",
            "MP-M": "0e+1o",
            "MP-L": "0e+1o+2e",
            "OFF-S": "0e",
            "OFF-M": "0e+1o",
            "OFF-L": "0e+1o+2e",
        }[model_size],
    ),
    offsets=np.zeros(num_species),
    cutoff=5.0,
    epsilon=1 / avg_num_neighbors,
    skip_connection_first_layer=("MP" in model_size),
    replicate_original_group=False,
)

# Dummy data
vecs = jax.random.normal(jax.random.key(0), (num_edges, 3))
species = jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_species)
senders, receivers = jax.random.randint(
    jax.random.key(0), (2, num_edges), 0, num_atoms
)
graph_index = jax.random.randint(jax.random.key(0), (num_atoms,), 0, num_graphs)
graph_index = jnp.sort(graph_index)

target_E = jax.random.normal(jax.random.key(0), (num_graphs,))
target_F = jax.random.normal(jax.random.key(0), (num_atoms, 3))
target_V = jax.random.normal(jax.random.key(0), (num_graphs, 3, 3))

nats = jnp.zeros((num_graphs,), dtype=jnp.int32).at[graph_index].add(1)
mask = jnp.ones((num_edges,), dtype=bool)

batch_dict = dict(
    nn_vecs=vecs,
    species=species,
    inda=senders,
    indb=receivers,
    inde=graph_index,
    nats=nats,
    mask=mask,
)

# Initialization
w = jax.jit(model.init)(jax.random.key(0), batch_dict)

E, F, V = model.apply(w, batch_dict)

assert E.shape == (num_graphs,)
assert F.shape == (num_atoms, 3)
assert V.shape == (num_graphs, 3, 3)