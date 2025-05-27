import time

import cuequivariance as cue
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ase.io import read
from jax import config

from mace_jax.data.neighborhood import get_neighborhood
from mace_jax.data.utils import get_edge_vectors_and_lengths
from mace_jax.modules.loss import energy_mse, force_mse, stress_mse
from mace_jax.modules.models import MACEModel
from mace_jax.tools.plot_train import plot_training_results

config.update("jax_enable_x64", True)

dataset = read("../data/Cu_dataset.xyz", index=":")
cutoff = 6.0
energy_list = []
stress_list = []
forces_list = []
senders_list = []
receivers_list = []
cell_list = []
edge_vectors_list = []
edge_index_list = []
nats_list = []
batch_offset = 0
for i, atoms in enumerate(dataset):
    edge_index, shifts, unit_shifts, cell = get_neighborhood(
        atoms.positions,
        cutoff=cutoff,
        pbc=atoms.pbc,
        cell=atoms.cell.array,
    )

    energy = jnp.array(atoms.get_potential_energy())
    stress = jnp.array(atoms.get_stress(voigt=False))
    forces = jnp.array(atoms.get_forces())
    positions = jnp.array(atoms.get_positions())
    edge_index = jnp.array(edge_index, dtype=jnp.int64)

    edge_vectors, distances = get_edge_vectors_and_lengths(
        positions,
        senders=edge_index[0],
        receivers=edge_index[1],
        shifts=shifts,
        normalize=False,
    )
    energy_list.append(energy)
    stress_list.append(stress)
    forces_list.append(forces)
    edge_vectors_list.append(edge_vectors)
    edge_index_list.append(edge_index + batch_offset)
    cell_list.append(cell)
    nats_list.append(len(atoms))
    batch_offset += len(atoms)

num_species = 1
edge_vectors = jnp.concatenate(edge_vectors_list, axis=0)
edge_index = jnp.concatenate(edge_index_list, axis=1)
nats = jnp.array(nats_list)
num_edges = len(edge_vectors)
num_nodes = jnp.sum(nats)
num_graphs = len(dataset)

senders = edge_index[0]
receivers = edge_index[1]
cell = jnp.stack(cell_list)

forces = jnp.concatenate(forces_list)
energy = jnp.stack(energy_list)
stress = jnp.stack(stress_list)

graph_index = jnp.concatenate(
    [jnp.zeros(nats[i], dtype=jnp.int64) + i for i in range(num_graphs)]
)

mask = jnp.ones((num_edges,), dtype=bool)
species = jnp.concatenate(
    [jnp.ones(len(forces), dtype=jnp.int64) * i for i in range(num_species)]
)

assert len(senders) == num_edges
assert len(receivers) == num_edges
assert cell.shape == (num_graphs, 3, 3)
assert len(graph_index) == num_nodes
assert len(energy) == num_graphs
assert len(forces) == num_nodes
assert stress.shape == (num_graphs, 3, 3)
assert len(species) == num_nodes

batch_dict = dict(
    nn_vecs=edge_vectors,
    species=species,
    inda=senders,
    indb=receivers,
    inde=graph_index,
    nats=nats,
    mask=mask,
    cell=cell,
)
model_size = "MP-S"
avg_num_neighbors = jnp.int64(num_edges / num_nodes)

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
    cutoff=cutoff,
    epsilon=float(1.0 / avg_num_neighbors),
    skip_connection_first_layer=("MP" in model_size),
    replicate_original_group=False,
    num_polynomial_cutoff=6,
    atomic_numbers=jnp.array([29]),
    pair_repulsion=True,
)

# Initialization
model_weights = jax.jit(model.init)(jax.random.key(0), batch_dict)
opt = optax.adam(5e-3)
model_opt_state = opt.init(model_weights)
step_count = 0
num_steps = 400


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
        E_mse = energy_mse(E, target_E)
        F_mse = force_mse(F, target_F)
        S_mse = stress_mse(V, target_V, batch_dict["cell"])
        loss = E_mse + 100 * F_mse + 10 * S_mse
        return loss, (E_mse, F_mse, S_mse, loss)

    grad, (E_mse, F_mse, S_mse, loss) = jax.grad(loss_fn, has_aux=True)(model_weights)
    updates, model_opt_state = opt.update(grad, model_opt_state)
    model_weights = optax.apply_updates(model_weights, updates)
    return model_weights, model_opt_state, (E_mse, F_mse, S_mse, loss)


target_E = energy
target_F = forces
target_S = stress

# compilation
E, F, V = model.apply(model_weights, batch_dict)

assert E.shape == target_E.shape
assert F.shape == target_F.shape
assert V.shape == target_S.shape

_ = step(model_weights, model_opt_state, batch_dict, target_E, target_F, target_S)

t0 = time.perf_counter()

training_losses = {"energy_loss": [], "force_loss": [], "stress_loss": []}

for _ in range(num_steps):
    step_count += 1

    (model_weights, model_opt_state, (E_mse, F_mse, S_mse, loss)) = step(
        model_weights, model_opt_state, batch_dict, target_E, target_F, target_S
    )

    training_losses["energy_loss"].append(float(E_mse))
    training_losses["force_loss"].append(float(F_mse))
    training_losses["stress_loss"].append(float(S_mse))

    print(
        f"Step {step_count}, "
        f"energy_mse: {E_mse:.4f}, "
        f"force_mse: {F_mse:.4f}, "
        f"stress_mse: {S_mse:.4f}, "
        f"loss: {loss:.4f}"
    )

    if step_count % 200 == 0:
        final_E, final_F, final_V = model.apply(model_weights, batch_dict)
        final_predictions = {
            "energy": final_E,
            "forces": final_F,
            "stress": final_V / jnp.linalg.det(batch_dict["cell"]).reshape(-1, 1, 1),
        }
        final_labels = {"energy": target_E, "forces": target_F, "stress": target_S}
        plot_training_results(
            training_losses, final_predictions, final_labels, step_count
        )

jax.block_until_ready(model_weights)
t1 = time.perf_counter()

runtime_per_step = 1e3 * (t1 - t0) / num_steps
print(f"{runtime_per_step:.0f} ms per step")

E, F, V = model.apply(model_weights, batch_dict)
print(f"E pred: {E}")
print(f"F pred: {F}")
print(f"S pred: {V / jnp.linalg.det(batch_dict['cell']).reshape(-1, 1, 1)}")

print(f"E true: {target_E}")
print(f"F true: {target_F}")
print(f"S true: {target_S}")
