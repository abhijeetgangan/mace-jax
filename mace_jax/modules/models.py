from typing import Callable

import cuequivariance as cue
import cuequivariance_jax as cuex
import flax
import flax.linen
import jax
import jax.numpy as jnp
import numpy as onp
from cuequivariance.group_theory.experimental.e3nn import O3_e3nn
from cuequivariance.group_theory.experimental.mace import symmetric_contraction
from cuequivariance_jax.experimental.utils import MultiLayerPerceptron

from .blocks import radial_basis


class MACELayer(flax.linen.Module):
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
        vectors: cuex.RepArray,  # [num_edges, 3]
        node_feats: cuex.RepArray,  # [num_nodes, irreps]
        node_species: jax.Array,  # [num_nodes] int between 0 and num_species-1
        radial_embeddings: jax.Array,  # [num_edges, radial_embedding_dim]
        senders: jax.Array,  # [num_edges]
        receivers: jax.Array,  # [num_edges]
    ):
        dtype = node_feats.dtype

        if self.last:
            hidden_out = self.hidden_irreps.filter(keep=self.output_irreps)
        else:
            hidden_out = self.hidden_irreps

        def lin(irreps: cue.Irreps, inp: cuex.RepArray, name: str):
            e = cue.descriptors.linear(inp.irreps, irreps)
            w = self.param(name, jax.random.normal, (e.inputs[0].irreps.dim,), dtype)
            return cuex.equivariant_polynomial(e, [w, inp], name=f"{self.name}_{name}")

        def linZ(irreps: cue.Irreps, inp: cuex.RepArray, name: str):
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


class MACE(flax.linen.Module):
    offsets: onp.ndarray
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

    @flax.linen.compact
    def __call__(
        self, batch: dict[str, jax.Array | int]
    ) -> tuple[jax.Array, jax.Array]:
        vecs: jax.Array = batch["nn_vecs"]  # [num_edges, 3]
        # [num_nodes] int between 0 and num_species-1
        species: jax.Array = batch["species"]
        senders: jax.Array = batch["inda"]  # [num_edges]
        receivers: jax.Array = batch["indb"]  # [num_edges]
        graph_index: jax.Array = batch["inde"]  # [num_nodes]
        num_graphs: int = jnp.shape(batch["nats"])[0]
        mask: jax.Array = batch["mask"]  # [num_edges]

        def model(vecs):
            with cue.assume(
                O3_e3nn if self.replicate_original_group else cue.O3, cue.ir_mul
            ):
                w = self.param(
                    "linear_embedding",
                    jax.random.normal,
                    (self.num_species, self.num_features),
                    vecs.dtype,
                )
                node_feats = cuex.as_irreps_array(
                    w[species] / jnp.sqrt(self.num_species)
                )

                radial_embeddings = jax.vmap(
                    radial_basis(self.cutoff, self.num_radial_basis)
                )(jnp.linalg.norm(vecs, axis=1))
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
                return jnp.sum(Es), Es

        Fterms, Ei = jax.grad(model, has_aux=True)(vecs)
        offsets = jnp.asarray(self.offsets, dtype=Ei.dtype)
        Ei = Ei + offsets[species]

        E = jnp.zeros((num_graphs,), Ei.dtype).at[graph_index].add(Ei)
        Fterms = jnp.where(jnp.expand_dims(mask, -1), Fterms, 0.0)

        nats = jnp.shape(species)[0]
        F = (
            jnp.zeros((nats, 3), Ei.dtype)
            .at[senders]
            .add(Fterms)
            .at[receivers]
            .add(-Fterms)
        )

        return E, F
