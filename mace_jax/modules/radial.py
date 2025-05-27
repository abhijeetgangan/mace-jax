import flax
import jax
import jax.numpy as jnp
from ase.data import covalent_radii

# ASE covalent radii data (in Angstroms)
COVALENT_RADII = jnp.array(covalent_radii)


class PolynomialCutoff(flax.linen.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.

    Parameters
    ----------
    r_max : float
        Cutoff radius
    p : int, default=6
        Polynomial order
    """

    r_max: float
    p: int = 6

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply polynomial cutoff function.

        Parameters
        ----------
        x : jax.Array
            Input distances, shape [num_edges]

        Returns
        -------
        jax.Array
            Cutoff values, shape [num_edges]
        """
        return self.calculate_envelope(x, self.r_max, self.p)

    @staticmethod
    def calculate_envelope(x: jax.Array, r_max: float, p: int) -> jax.Array:
        """Calculate polynomial envelope function.

        Parameters
        ----------
        x : jax.Array
            Input distances
        r_max : float
            Maximum cutoff radius
        p : int
            Polynomial order

        Returns
        -------
        jax.Array
            Envelope values
        """

        # Same as below
        '''
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
        '''

        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(r_over_r_max, p)
            + p * (p + 2.0) * jnp.power(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * jnp.power(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)


class radial_basis(flax.linen.Module):
    """Radial basis functions using Bessel functions with polynomial envelope.

    Parameters
    ----------
    r_max : float
        Cutoff radius for the radial basis functions.
    num_radial_basis : int
        Number of radial basis functions to use.
    num_polynomial_cutoff : int, default=6
        Number of polynomial cutoff to use.
    """

    r_max: float
    num_radial_basis: int
    num_polynomial_cutoff: int = 6

    def setup(self):
        self.cutoff_fn = PolynomialCutoff(
            r_max=self.r_max, p=self.num_polynomial_cutoff
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
    def __call__(self, edge_lengths: jax.Array) -> jax.Array:
        """Compute radial basis functions with cutoff.

        Parameters
        ----------
        edge_lengths : jax.Array
            Edge distances, shape [num_edges]

        Returns
        -------
        jax.Array
            Radial basis function values.
        """
        assert edge_lengths.ndim == 0
        cutoff = self.cutoff_fn(edge_lengths)
        radial = self.bessel(edge_lengths)
        return radial * cutoff


class ZBLBasis(flax.linen.Module):
    """Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.

    Parameters
    ----------
    p : int, default=6
        Polynomial order for cutoff
    trainable : bool, default=False
        Whether ZBL parameters are trainable
    """

    p: int = 6
    trainable: bool = False

    def setup(self):
        """Initialize ZBL parameters."""
        # ZBL potential coefficients
        self.c = jnp.array([0.1818, 0.5099, 0.2802, 0.02817])

        # ZBL screening parameters
        if self.trainable:
            self.a_exp = self.param("a_exp", lambda key: jnp.array(0.300))
            self.a_prefactor = self.param("a_prefactor", lambda key: jnp.array(0.4543))
        else:
            self.a_exp = 0.300
            self.a_prefactor = 0.4543

    def __call__(
        self,
        distances: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        atomic_numbers: jax.Array,
    ) -> jax.Array:
        """Compute ZBL pair repulsion energies.

        Parameters
        ----------
        distances : jax.Array
            Edge distances, shape [num_edges]
        species : jax.Array
            Node species indices, shape [num_nodes]
        senders : jax.Array
            Sender node indices, shape [num_edges]
        receivers : jax.Array
            Receiver node indices, shape [num_edges]
        atomic_numbers : jax.Array
            Atomic numbers for each species, shape [num_species]

        Returns
        -------
        jax.Array
            ZBL energies per node, shape [num_nodes]
        """
        # Get atomic numbers for sender and receiver nodes
        node_atomic_numbers = atomic_numbers[species]
        Z_u = node_atomic_numbers[senders]
        Z_v = node_atomic_numbers[receivers]

        # Calculate screening parameter
        a = (
            self.a_prefactor
            * 0.529  # Bohr radius in Angstroms
            / (jnp.power(Z_u, self.a_exp) + jnp.power(Z_v, self.a_exp))
        )

        r_over_a = distances / a

        # ZBL potential function
        phi = (
            self.c[0] * jnp.exp(-3.2 * r_over_a)
            + self.c[1] * jnp.exp(-0.9423 * r_over_a)
            + self.c[2] * jnp.exp(-0.4028 * r_over_a)
            + self.c[3] * jnp.exp(-0.2016 * r_over_a)
        )

        # Calculate ZBL energy for each edge
        v_edges = (14.3996 * Z_u * Z_v) / distances * phi

        # Apply polynomial cutoff based on covalent radii
        r_max = COVALENT_RADII[Z_u.astype(int)] + COVALENT_RADII[Z_v.astype(int)]
        envelope = PolynomialCutoff.calculate_envelope(distances, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope

        # Sum contributions to each node
        num_nodes = species.shape[0]
        V_ZBL = jnp.zeros((num_nodes,), dtype=v_edges.dtype).at[receivers].add(v_edges)

        return V_ZBL


class AgnesiTransform(flax.linen.Module):
    """Agnesi transform for radial distances.

    Parameters
    ----------
    q : float, default=0.9183
        Agnesi parameter q
    p : float, default=4.5791
        Agnesi parameter p
    a : float, default=1.0805
        Agnesi parameter a
    trainable : bool, default=False
        Whether parameters are trainable
    """

    q: float = 0.9183
    p: float = 4.5791
    a: float = 1.0805
    trainable: bool = False

    def setup(self):
        """Initialize Agnesi transform parameters."""
        if self.trainable:
            self.q_param = self.param("q", lambda key: jnp.array(self.q))
            self.p_param = self.param("p", lambda key: jnp.array(self.p))
            self.a_param = self.param("a", lambda key: jnp.array(self.a))
        else:
            self.q_param = self.q
            self.p_param = self.p
            self.a_param = self.a

    def __call__(
        self,
        x: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        atomic_numbers: jax.Array,
    ) -> jax.Array:
        """Apply Agnesi transform."""
        # Get atomic numbers for sender and receiver nodes
        node_atomic_numbers = atomic_numbers[species]
        Z_u = node_atomic_numbers[senders]
        Z_v = node_atomic_numbers[receivers]

        r_0 = 0.5 * (COVALENT_RADII[Z_u.astype(int)] + COVALENT_RADII[Z_v.astype(int)])
        r_over_r_0 = x / r_0

        numerator = self.a_param * jnp.power(r_over_r_0, self.q_param)
        denominator = 1 + jnp.power(r_over_r_0, self.q_param - self.p_param)

        return 1.0 / (1 + numerator / denominator)
