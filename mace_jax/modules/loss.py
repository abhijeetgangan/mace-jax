import jax.numpy as jnp


def energy_mse(E, target_E):
    return jnp.mean((E - target_E) ** 2)


def force_mse(F, target_F):
    return jnp.mean((F - target_F) ** 2)


def virial_mse(V, target_V):
    return jnp.mean((V - target_V) ** 2)
