import time

import cuequivariance as cue
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mace_jax.modules.loss import energy_mse, force_mse, virial_mse
from mace_jax.modules.models import MACEModel

# Dataset specifications
num_species = 50
num_graphs = 100
avg_num_neighbors = 20

model_size = "MP-M"

if "MP" in model_size:
    num_atoms = 3_000
    num_edges = 160_000
else:
    num_atoms = 4_000
    num_edges = 70_000

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
senders, receivers = jax.random.randint(jax.random.key(0), (2, num_edges), 0, num_atoms)
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
model_weights = jax.jit(model.init)(jax.random.key(0), batch_dict)
opt = optax.adam(1e-2)
model_opt_state = opt.init(model_weights)
step_count = 0


# Training
@jax.jit
def step(
    model_weights: dict,
    model_opt_state: optax.OptState,
    batch_dict: dict,
    target_E: jax.Array,
    target_F: jax.Array,
    target_V: jax.Array,
):

    def loss_fn(w):
        E, F, V = model.apply(w, batch_dict)
        E_loss = energy_mse(E, target_E)
        F_loss = force_mse(F, target_F)
        V_loss = virial_mse(V, target_V)
        return E_loss + F_loss + V_loss, (E_loss, F_loss, V_loss)

    grad, (E_loss, F_loss, V_loss) = jax.grad(loss_fn, has_aux=True)(model_weights)
    updates, model_opt_state = opt.update(grad, model_opt_state)
    model_weights = optax.apply_updates(model_weights, updates)
    return model_weights, model_opt_state, (E_loss, F_loss, V_loss)


# compilation
_ = step(model_weights, model_opt_state, batch_dict, target_E, target_F, target_V)

t0 = time.perf_counter()

for _ in range(10):
    (model_weights, model_opt_state, (E_loss, F_loss, V_loss)) = step(
        model_weights, model_opt_state, batch_dict, target_E, target_F, target_V
    )

    print(
        f"Step {step_count}, "
        f"energy_mse: {E_loss:.4f}, "
        f"force_mse: {F_loss:.4f}, "
        f"virial_mse: {V_loss:.4f}"
    )

    step_count += 1

jax.block_until_ready(model_weights)
t1 = time.perf_counter()

runtime_per_step = 1e3 * (t1 - t0) / 10
print(f"{runtime_per_step:.0f} ms per step")
