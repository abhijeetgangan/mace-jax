import flax
import flax.linen
import jax
import jax.numpy as jnp


# Just Bessel for now
class radial_basis(flax.linen.Module):
    """Radial basis functions using Bessel functions with polynomial envelope.

    Parameters
    ----------
    r_max : float
        Cutoff radius for the radial basis functions.
    num_radial_basis : int
        Number of radial basis functions to use.
    num_polynomial_cutoff : int, default=5
        Order of the polynomial used for the envelope function.
    """

    r_max: float
    num_radial_basis: int
    num_polynomial_cutoff: int = 5

    def envelope(self, x: jax.Array) -> jax.Array:
        """Polynomial envelope function that smoothly goes to zero at r_max.

        Parameters
        ----------
        x : jax.Array
            Input distances.

        Returns
        -------
        jax.Array
            Envelope function values.
        """
        p = float(self.num_polynomial_cutoff)
        xs = x / self.r_max
        xp = jnp.power(xs, self.num_polynomial_cutoff)
        return (
            1.0
            - 0.5 * (p + 1.0) * (p + 2.0) * xp
            + p * (p + 2.0) * xp * xs
            - 0.5 * p * (p + 1.0) * xp * xs * xs
        )

    def bessel(self, x: jax.Array) -> jax.Array:
        """Bessel basis functions.

        Parameters
        ----------
        x : jax.Array
            Input distances.

        Returns
        -------
        jax.Array
            Bessel function values.
        """
        n = jnp.arange(1, self.num_radial_basis + 1, dtype=x.dtype)
        return (
            jnp.sqrt(2.0 / self.r_max)
            * jnp.pi
            * n
            / self.r_max
            * jnp.sinc(n * x / self.r_max)
        )

    @flax.linen.compact
    def __call__(self, edge: jax.Array) -> jax.Array:
        """Compute radial basis functions with cutoff.

        Parameters
        ----------
        edge : jax.Array
            Edge distance.

        Returns
        -------
        jax.Array
            Radial basis function values.
        """
        assert edge.ndim == 0
        cutoff = jnp.where(edge < self.r_max, self.envelope(edge), 0.0)
        radial = self.bessel(edge)
        return radial * cutoff
