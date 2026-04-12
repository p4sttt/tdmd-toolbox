from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax.typing import ArrayLike


from tdmd.core.tensor_product import LinearTransform
from tdmd.core.decomposition import _truncated_tsvd_impl, _validate_threshold


class TDMDResult(NamedTuple):
    modes: jax.Array
    schur_tensor: jax.Array


class TDMDIIResult(NamedTuple):
    modes: jax.Array
    schur_tensor: jax.Array
    amplitudes: jax.Array
    multirank: jax.Array


def _invert_transformed_singular_values(S_hat: jax.Array, signvals_threshold: float) -> jax.Array:
    singular_values = jnp.diagonal(S_hat, axis1=1, axis2=2)
    keep = jnp.abs(singular_values) > signvals_threshold
    safe_singular_values = jnp.where(keep, singular_values, 1)
    singular_values_inv = jnp.where(keep, 1 / safe_singular_values, 0)

    eye = jnp.eye(S_hat.shape[1], dtype=S_hat.dtype)[None, :, :]
    return singular_values_inv[:, :, None] * eye


def _truncate_tsvdii_impl(A_hat: jax.Array, gamma: float) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    U_hat, singular_values, Vh_hat = jnp.linalg.svd(A_hat, full_matrices=False)

    energies = (jnp.abs(singular_values) ** 2).reshape(-1)
    sorted_energies = jnp.sort(energies)[::-1]
    cumulative_energy = jnp.cumsum(sorted_energies)
    keep_count = jnp.searchsorted(cumulative_energy, gamma * cumulative_energy[-1], side="left") + 1
    cutoff = sorted_energies[keep_count - 1]

    keep = (jnp.abs(singular_values) ** 2) >= cutoff
    masked_singular_values = jnp.where(keep, singular_values, 0)
    multirank = jnp.sum(keep, axis=1)

    return U_hat, masked_singular_values, Vh_hat, multirank


def _compact_face_factors(
    U_hat: jax.Array,
    singular_values: jax.Array,
    Vh_hat: jax.Array,
    multirank: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    kmax = singular_values.shape[1]
    keep_cols = jnp.arange(kmax)[None, :] < multirank[:, None]

    U_hat = jnp.where(keep_cols[:, None, :], U_hat, 0)
    singular_values = jnp.where(keep_cols, singular_values, 0)
    Vh_hat = jnp.where(keep_cols[:, :, None], Vh_hat, 0)

    eye = jnp.eye(kmax, dtype=singular_values.dtype)[None, :, :]
    S_hat = singular_values[:, :, None] * eye
    return U_hat, S_hat, Vh_hat


@partial(jax.jit, static_argnames=("svd_threshold", "signvals_threshold"))
def _tdmd_impl(
    X: ArrayLike,
    Y: ArrayLike,
    L: LinearTransform,
    svd_threshold: float,
    signvals_threshold: float,
) -> TDMDResult:
    U_hat, S_hat, Vh_hat = _truncated_tsvd_impl(X, L, svd_threshold)
    Y_hat = L.to_slices(Y)

    Uh_hat = jnp.swapaxes(U_hat.conj(), 1, 2)
    V_hat = jnp.swapaxes(Vh_hat.conj(), 1, 2)
    S_hat_inv = _invert_transformed_singular_values(S_hat, signvals_threshold)

    A_tilde_hat = Uh_hat @ Y_hat @ V_hat @ S_hat_inv
    T_hat, W_hat = jsc.linalg.schur(A_tilde_hat)
    modes_hat = U_hat @ W_hat

    return TDMDResult(L.from_slices(modes_hat), L.from_slices(T_hat))


@partial(jax.jit, static_argnames=("gamma", "signvals_threshold"))
def _tdmdii_impl(
    X: ArrayLike,
    Y: ArrayLike,
    L: LinearTransform,
    gamma: float,
    signvals_threshold: float,
) -> TDMDIIResult:
    X_hat = L.to_slices(X)
    Y_hat = L.to_slices(Y)

    U_hat, singular_values, Vh_hat, multirank = _truncate_tsvdii_impl(X_hat, gamma)
    U_hat, S_hat, Vh_hat = _compact_face_factors(U_hat, singular_values, Vh_hat, multirank)

    Uh_hat = jnp.swapaxes(U_hat.conj(), 1, 2)
    V_hat = jnp.swapaxes(Vh_hat.conj(), 1, 2)
    S_hat_inv = _invert_transformed_singular_values(S_hat, signvals_threshold)

    A_tilde_hat = Uh_hat @ Y_hat @ V_hat @ S_hat_inv
    T_hat, W_hat = jsc.linalg.schur(A_tilde_hat)
    modes_hat = U_hat @ W_hat

    X0_hat = L.to_slices(X[:, :1, :])
    amplitudes_hat = jnp.swapaxes(modes_hat.conj(), 1, 2) @ X0_hat

    return TDMDIIResult(
        L.from_slices(modes_hat),
        L.from_slices(T_hat),
        L.from_slices(amplitudes_hat),
        multirank,
    )


def tdmd(
    X: ArrayLike,
    Y: ArrayLike,
    L: LinearTransform,
    svd_threshold: float = 0.0,
    signvals_threshold: float = 0.0,
) -> TDMDResult:
    """Compute tensor DMD modes and eigenvalues under the transform ``L``.

    Args:
        X: Snapshot tensor with shape ``(m, n, k)``.
        Y: Time-shifted snapshot tensor aligned with ``X``.
        L: Linear transform defining the tensor product algebra.
        svd_threshold: Absolute tube-spectrum cutoff used to truncate the tensor SVD
            of ``X`` before forming the reduced operator.
        signvals_threshold: Absolute cutoff applied when inverting transformed
            singular values. Singular values not greater than this threshold
            are treated as zero in the pseudo-inverse.

    Returns:
        A tuple ``(modes, schur_tensor)`` where ``modes`` contains the tensor
        DMD modes and ``schur_tensor`` contains the tensor Schur form of the
        reduced operator.
    """
    svd_threshold = _validate_threshold(svd_threshold)
    signvals_threshold = _validate_threshold(signvals_threshold)
    return _tdmd_impl(X, Y, L, svd_threshold, signvals_threshold)


def tdmdii(
    X: ArrayLike,
    Y: ArrayLike,
    L: LinearTransform,
    gamma: float = 0.99999,
    signvals_threshold: float = 0.0,
) -> TDMDIIResult:
    """Compute the multirank TDMDII variant under transform ``L``.

    This variant follows the tr-tSVDMII truncation strategy from the
    ``star_M``-DMDII framework, keeping transformed singular values globally
    across faces until the retained squared-energy fraction reaches ``gamma``.

    Args:
        X: Snapshot tensor with shape ``(m, n, k)``.
        Y: Time-shifted snapshot tensor aligned with ``X``.
        L: Linear transform defining the tensor product algebra.
        gamma: Energy-retention level in ``(0, 1]`` used for the multirank
            truncation in the transform domain.
        signvals_threshold: Absolute cutoff applied when inverting retained
            transformed singular values.

    Returns:
        A tuple ``(modes, schur_tensor, amplitudes, multirank)`` where
        ``multirank`` stores the retained rank of each transformed face.
    """
    gamma = float(gamma)
    if gamma <= 0 or gamma > 1 or not jnp.isfinite(gamma):
        raise ValueError(f"Expected gamma in (0, 1]; got {gamma}.")

    signvals_threshold = _validate_threshold(signvals_threshold)
    return _tdmdii_impl(X, Y, L, gamma, signvals_threshold)
