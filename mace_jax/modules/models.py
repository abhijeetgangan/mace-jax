from typing import Callable

import cuequivariance as cue
import cuequivariance_jax as cuex
import flax
import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
from cuequivariance.group_theory.experimental.e3nn import O3_e3nn
from cuequivariance.group_theory.experimental.mace import symmetric_contraction
from cuequivariance_jax.experimental.utils import MultiLayerPerceptron

from mace_jax.modules.radial import (
    AgnesiTransform,
    PolynomialCutoff,
    ZBLBasis,
    radial_basis,
)


class MACELayer(flax.linen.Module):
    """MACE equivariant layer.

    Parameters
    ----------
    first : bool
        Whether this is the first layer in the network.
    last : bool
        Whether this is the last layer in the network.
    num_species : int
        Number of atomic species in the dataset.
    num_features : int
        Number of features in the network, typically 128.
    interaction_irreps : cue.Irreps
        Irreducible representations for interactions, typically 0e+1o+2e+3o.
    hidden_irreps : cue.Irreps
        Irreducible representations for hidden layers, typically 0e+1o.
    activation : Callable
        Activation function, typically silu.
    epsilon : float
        Small constant for numerical stability, typically 1/avg_num_neighbors.
    max_ell : int
        Maximum angular momentum, typically 3.
    correlation : int
        Order of correlation, typically 3.
    output_irreps : cue.Irreps
        Irreducible representations for output, typically 1x0e.
    readout_mlp_irreps : cue.Irreps
        Irreducible representations for readout MLP, typically 16x0e.
    replicate_original_mace_sc : bool, default=True
        Whether to replicate the original MACE skip connection behavior.
    skip_connection_first_layer : bool, default=False
        Whether to use skip connections in the first layer.
    """

    first: bool
    last: bool
    num_species: int
    num_features: int  # typically 128
    interaction_irreps: cue.Irreps  # typically 0e+1o+2e+3o
    hidden_irreps: cue.Irreps  # typically 0e+1o
    activation: Callable  # typically silu
    epsilon: float  # typically 1/avg_num_neighbors
    max_ell: int  # typically 3
    correlation: int  # typically 3
    output_irreps: cue.Irreps  # typically 1x0e
    readout_mlp_irreps: cue.Irreps  # typically 16x0e
    replicate_original_mace_sc: bool = True
    skip_connection_first_layer: bool = False

    @flax.linen.compact
    def __call__(
        self,
        vectors: cuex.RepArray,
        node_feats: cuex.RepArray,
        node_species: jax.Array,
        radial_embeddings: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ):
        """Apply the MACE layer to the input data.

        Parameters
        ----------
        vectors : cuex.RepArray
            Edge vectors, shape [num_edges, 3]
        node_feats : cuex.RepArray
            Node features, shape [num_nodes, irreps]
        node_species : jax.Array
            Node atomic species, shape [num_nodes], integers between 0 and num_species-1
        radial_embeddings : jax.Array
            Radial embeddings for each edge, shape [num_edges, radial_embedding_dim]
        senders : jax.Array
            Indices of sender nodes for each edge, shape [num_edges]
        receivers : jax.Array
            Indices of receiver nodes for each edge, shape [num_edges]

        Returns
        -------
        cuex.RepArray
            Updated node features
        """
        dtype = node_feats.dtype

        if self.last:
            hidden_out = self.hidden_irreps.filter(keep=self.output_irreps)
        else:
            hidden_out = self.hidden_irreps

        def lin(irreps: cue.Irreps, inp: cuex.RepArray, name: str):
            """Linear layer with shared weights.

            Parameters
            ----------
            irreps : cue.Irreps
                Output irreducible representations
            inp : cuex.RepArray
                Input equivariant features
            name : str
                Name of the parameter

            Returns
            -------
            cuex.RepArray
                Output of the linear layer with shared weights
            """
            e = cue.descriptors.linear(inp.irreps, irreps)
            w = self.param(name, jax.random.normal, (e.inputs[0].irreps.dim,), dtype)
            return cuex.equivariant_polynomial(e, [w, inp], name=f"{self.name}_{name}")

        def linZ(irreps: cue.Irreps, inp: cuex.RepArray, name: str):
            """Linear layer with species-dependent weights.

            Parameters
            ----------
            irreps : cue.Irreps
                Output irreducible representations
            inp : cuex.RepArray
                Input equivariant features
            name : str
                Name of the parameter

            Returns
            -------
            cuex.RepArray
                Output of the linear layer with species-dependent weights
            """
            # Dividing by num_species for consistency with the 1-hot implementation
            e = cue.descriptors.linear(inp.irreps, irreps)
            e = e * (1.0 / self.num_species**0.5)
            w = self.param(
                name,
                jax.random.normal,
                (self.num_species, e.inputs[0].irreps.dim),
                dtype,
            )
            return cuex.equivariant_polynomial(
                e,
                [w, inp],
                indices=[node_species, None, None],
                name=f"{self.name}_{name}",
            )

        def conv(
            node_features: cuex.RepArray,
            sph: cuex.RepArray,
            radial_embeddings: jax.Array,
            senders: jax.Array,
            receivers: jax.Array,
        ) -> cuex.RepArray:
            """Perform convolution operation on node features.

            Parameters
            ----------
            node_features : cuex.RepArray
                Input node features
            sph : cuex.RepArray
                Spherical harmonics representation of edge vectors
            radial_embeddings : jax.Array
                Radial embeddings for each edge
            senders : jax.Array
                Indices of sender nodes for each edge
            receivers : jax.Array
                Indices of receiver nodes for each edge

            Returns
            -------
            cuex.RepArray
                Updated node features after convolution
            """
            descriptor = cue.descriptors.channelwise_tensor_product(
                node_features.irreps, sph.irreps, self.interaction_irreps
            )
            descriptor = descriptor.squeeze_modes().flatten_coefficient_modes()
            descriptor = descriptor * self.epsilon

            w = MultiLayerPerceptron(
                [64, 64, 64, descriptor.inputs[0].dim],
                self.activation,
                output_activation=False,
                with_bias=False,
            )(radial_embeddings)

            node_features = cuex.equivariant_polynomial(
                descriptor,
                [w, node_features, sph],
                outputs_shape_dtype=jax.ShapeDtypeStruct(
                    (node_features.shape[0], -1), dtype
                ),
                indices=[None, senders, None, receivers],
                name=f"{self.name}_TP",
            )
            return node_features

        def sc(node_feats: cuex.RepArray) -> cuex.RepArray:
            """Perform symmetric contraction on node features.

            Parameters
            ----------
            node_feats : cuex.RepArray
                Input node features to be contracted

            Returns
            -------
            cuex.RepArray
                Node features after symmetric contraction
            """
            e, projection = symmetric_contraction(
                node_feats.irreps,
                self.num_features * hidden_out,
                range(1, self.correlation + 1),
            )
            projection = jnp.array(projection, dtype=dtype)
            n = projection.shape[0 if self.replicate_original_mace_sc else 1]
            w = self.param(
                "symmetric_contraction",
                jax.random.normal,
                (self.num_species, n, self.num_features),
                dtype,
            )
            if self.replicate_original_mace_sc:
                w = jnp.einsum("zau,ab->zbu", w, projection)
            w = jnp.reshape(w, (self.num_species, -1))

            return cuex.equivariant_polynomial(
                e,
                [w, node_feats],
                indices=[node_species, None, None],
                name=f"{self.name}_SC",
            )

        sph = cuex.spherical_harmonics(range(self.max_ell + 1), vectors)

        self_connection = None
        if not self.first or self.skip_connection_first_layer:
            self_connection = linZ(
                self.num_features * hidden_out, node_feats, "linZ_skip_tp"
            )
        node_feats = lin(node_feats.irreps, node_feats, "linear_up")
        node_feats = conv(node_feats, sph, radial_embeddings, senders, receivers)
        node_feats = lin(
            self.num_features * self.interaction_irreps, node_feats, "linear_down"
        )

        # This is only used in the first layer if it has no skip connection
        if self.first and not self.skip_connection_first_layer:
            # Selector TensorProduct
            node_feats = linZ(
                self.num_features * self.interaction_irreps,
                node_feats,
                "linZ_skip_tp_first",
            )

        node_feats = sc(node_feats)
        node_feats = lin(self.num_features * hidden_out, node_feats, "linear_post_sc")

        if self_connection is not None:
            node_feats = (
                node_feats + self_connection
            )  # [num_nodes, num_features * hidden_out]

        node_outputs = node_feats
        if self.last:  # Non linear readout for last layer
            assert self.readout_mlp_irreps.is_scalar()
            assert self.output_irreps.is_scalar()
            node_outputs = cuex.scalar_activation(
                lin(self.readout_mlp_irreps, node_outputs, "linear_mlp_readout"),
                self.activation,
            )
        node_outputs = lin(self.output_irreps, node_outputs, "linear_readout")

        return node_outputs, node_feats


class MACEModel(flax.linen.Module):
    """MACE model for molecular property prediction.

    Parameters
    ----------
    offsets : numpy.ndarray
        Atomic energy offsets for each species.
    num_species : int
        Number of atomic species in the dataset.
    cutoff : float
        Cutoff radius for neighbor interactions.
    num_layers : int
        Number of MACE layers in the model.
    num_features : int
        Number of features in the network.
    interaction_irreps : cue.Irreps
        Irreducible representations for interactions.
    hidden_irreps : cue.Irreps
        Irreducible representations for hidden layers.
    max_ell : int
        Maximum angular momentum.
    correlation : int
        Order of correlation.
    num_radial_basis : int
        Number of radial basis functions.
    epsilon : float
        Small constant for numerical stability.
    skip_connection_first_layer : bool
        Whether to use skip connections in the first layer.
    replicate_original_group : bool
        Whether to replicate the original MACE group behavior.
    atomic_numbers : jax.Array
        Atomic numbers for each species.
    pair_repulsion : bool
        Whether to use pair repulsion.
    num_polynomial_cutoff : int
        Number of polynomial cutoff to use.
    """

    offsets: np.ndarray
    num_species: int
    cutoff: float
    num_layers: int
    num_features: int
    interaction_irreps: cue.Irreps
    hidden_irreps: cue.Irreps
    max_ell: int
    correlation: int
    num_radial_basis: int
    epsilon: float
    skip_connection_first_layer: bool
    replicate_original_group: bool

    # ZBL parameters
    atomic_numbers: jax.Array
    pair_repulsion: bool = False
    num_polynomial_cutoff: int = 6

    def setup(self):
        """Setup the model."""
        # Pair repulsion
        if self.pair_repulsion:
            self.zbl_basis = ZBLBasis(p=self.num_polynomial_cutoff)

    @flax.linen.compact
    def __call__(
        self, batch: dict[str, jax.Array | int]
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass of the MACE model.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data with keys:
            - nn_vecs: Edge vectors, shape [num_edges, 3]
            - species: Node atomic species, shape [num_nodes], integers between 0 and num_species-1
            - inda: Indices of sender nodes for each edge, shape [num_edges]
            - indb: Indices of receiver nodes for each edge, shape [num_edges]
            - inde: Graph indices for each node, shape [num_nodes]
            - nats: Number of atoms in each graph, shape [num_graphs]
            - mask: Edge mask, shape [num_edges]

        Returns
        -------
        tuple
            - E: Total energy for each graph, shape [num_graphs]
            - F: Forces for each atom, shape [num_nodes, 3]
            - S: Stress tensor for each graph, shape [num_graphs, 3, 3]
        """
        vecs: jax.Array = batch["nn_vecs"]  # [num_edges, 3]
        # [num_nodes] int between 0 and num_species-1
        species: jax.Array = batch["species"]
        senders: jax.Array = batch["inda"]  # [num_edges]
        receivers: jax.Array = batch["indb"]  # [num_edges]
        graph_index: jax.Array = batch["inde"]  # [num_nodes]
        num_graphs: int = jnp.shape(batch["nats"])[0]
        mask: jax.Array = batch["mask"]  # [num_edges]

        def model(vecs):
            """Forward computation of the MACE model.

            Parameters
            ----------
            vecs : jax.Array
                Edge vectors, shape [num_edges, 3]

            Returns
            -------
            tuple
                - total_energy: Sum of atomic energies, scalar
                - atomic_energies: Energy contribution of each atom, shape [num_nodes]
            """
            with cue.assume(
                O3_e3nn if self.replicate_original_group else cue.O3, cue.ir_mul
            ):
                w = self.param(
                    "linear_embedding",
                    jax.random.normal,
                    (self.num_species, self.num_features),
                    vecs.dtype,
                )
                lengths = jnp.linalg.norm(vecs, axis=1)

                node_feats = cuex.as_irreps_array(
                    w[species] / jnp.sqrt(self.num_species)
                )

                radial_embeddings = jax.vmap(
                    radial_basis(
                        r_max=self.cutoff,
                        num_radial_basis=self.num_radial_basis,
                        num_polynomial_cutoff=self.num_polynomial_cutoff,
                    ),
                )(lengths)
                vecs = cuex.RepArray("1o", vecs)

                Es = 0
                for i in range(self.num_layers):
                    first = i == 0
                    last = i == self.num_layers - 1
                    output, node_feats = MACELayer(
                        first=first,
                        last=last,
                        num_species=self.num_species,
                        num_features=self.num_features,
                        interaction_irreps=self.interaction_irreps,
                        hidden_irreps=self.hidden_irreps,
                        activation=jax.nn.silu,
                        epsilon=self.epsilon,
                        max_ell=self.max_ell,
                        correlation=self.correlation,
                        output_irreps=cue.Irreps("1x0e"),
                        readout_mlp_irreps=cue.Irreps("16x0e"),
                        skip_connection_first_layer=self.skip_connection_first_layer,
                        name=f"layer_{i}",
                    )(vecs, node_feats, species, radial_embeddings, senders, receivers)
                    Es += jnp.squeeze(output.array, 1)

                # Add ZBL repulsion if enabled
                if self.pair_repulsion:
                    zbl_energies = self.zbl_basis(
                        distances=lengths,
                        species=species,
                        senders=senders,
                        receivers=receivers,
                        atomic_numbers=jnp.asarray(self.atomic_numbers),
                    )
                    Es += zbl_energies

                return jnp.sum(Es), Es

        edge_forces, Ei = jax.grad(model, has_aux=True)(vecs)
        offsets = jnp.asarray(self.offsets, dtype=Ei.dtype)
        Ei = Ei + offsets[species]

        E = jnp.zeros((num_graphs,), Ei.dtype).at[graph_index].add(Ei)
        edge_forces = jnp.where(jnp.expand_dims(mask, -1), edge_forces, 0.0)

        nats = jnp.shape(species)[0]
        F = (
            jnp.zeros((nats, 3), Ei.dtype)
            .at[senders]
            .add(edge_forces)
            .at[receivers]
            .add(-edge_forces)
        )

        # Compute virial
        edge_virial = jnp.einsum("zi,zj->zij", edge_forces, vecs)
        atom_virial_sender = (
            jnp.zeros((nats, 3, 3), Ei.dtype).at[senders].add(edge_virial)
        )
        atom_virial_receiver = (
            jnp.zeros((nats, 3, 3), Ei.dtype).at[receivers].add(edge_virial)
        )
        atom_virial = (atom_virial_sender + atom_virial_receiver) / 2
        atom_virial = (atom_virial + atom_virial.transpose(0, 2, 1)) / 2

        virial = jnp.zeros((num_graphs, 3, 3), Ei.dtype)
        V = virial.at[graph_index].add(atom_virial.sum(0))

        return E, F, V
