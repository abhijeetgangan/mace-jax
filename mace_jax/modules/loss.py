import jax.numpy as jnp


def energy_mse(energy: jnp.ndarray, target_energy: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((energy - target_energy) ** 2)


def force_mse(force: jnp.ndarray, target_force: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((force - target_force) ** 2)


def stress_mse(
    virial: jnp.ndarray, target_S: jnp.ndarray, cell: jnp.ndarray
) -> jnp.ndarray:
    volumes = jnp.linalg.det(cell)
    stress = virial / volumes.reshape(-1, 1, 1)
    stress = jnp.where(stress < 1e10, stress, jnp.zeros_like(stress))
    return jnp.mean((stress - target_S) ** 2)


def virial_mse(virial: jnp.ndarray, target_V: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((virial - target_V) ** 2)
