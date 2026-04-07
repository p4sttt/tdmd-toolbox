from functools import partial

import jax
import jax.numpy as jnp

from tdmd.core.decomposition import (
    _resolve_rank_from_spectrum,
    _truncated_tsvd_impl,
    _validate_truncation_policy,
    tensor_singular_spectrum,
)
from tdmd.core.tensor_product import LinearTransform, star_prod


def _validate_matrix_dmd_inputs(
    X: jax.Array,
    Y: jax.Array,
    rank: int | None,
    energy_threshold: float | None,
    svd_threshold: float,
) -> None:
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"dmd expects matrices X and Y; got ndim={X.ndim} and ndim={Y.ndim}.")
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape; got {X.shape} and {Y.shape}.")
    if min(X.shape) == 0:
        raise ValueError(f"X and Y must have positive dimensions; got shape {X.shape}.")
    if rank is not None:
        max_rank = min(X.shape)
        if rank < 1 or rank > max_rank:
            raise ValueError(
                f"rank must be in [1, {max_rank}] for X with shape {X.shape}; got {rank}."
            )
    _validate_truncation_policy(rank, energy_threshold, svd_threshold)


@partial(jax.jit, static_argnames=("rank",))
def _dmd_impl(X: jax.Array, Y: jax.Array, rank: int) -> tuple[jax.Array, jax.Array]:
    U, S, Vh = jnp.linalg.svd(X, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh.conj().T[:, :rank]

    sigma_r_inv = jnp.diag(1.0 / S_r)
    A_tilde = U_r.conj().T @ Y @ V_r @ sigma_r_inv
    eigenvalues, W = jnp.linalg.eig(A_tilde)
    modes = Y @ V_r @ sigma_r_inv @ W
    return modes, eigenvalues


def dmd(
    X: jax.Array,
    Y: jax.Array,
    rank: int | None = None,
    *,
    energy_threshold: float | None = None,
    svd_threshold: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute Dynamic Mode Decomposition modes and eigenvalues.

    Args:
        X: Snapshot matrix whose columns contain the input states.
        Y: Snapshot matrix containing the time-shifted states aligned with ``X``.
        rank: Optional truncation rank for the SVD-based reduced operator.
        energy_threshold: Optional fraction of spectral energy to retain.
        svd_threshold: Absolute singular-value cutoff applied before inversion.

    Returns:
        A tuple ``(modes, eigenvalues)`` where ``modes`` contains the DMD modes
        and ``eigenvalues`` contains the eigenvalues of the reduced operator.
    """
    _validate_matrix_dmd_inputs(X, Y, rank, energy_threshold, svd_threshold)
    spectrum = jnp.linalg.svd(X, full_matrices=False, compute_uv=False)
    resolved_rank = _resolve_rank_from_spectrum(
        spectrum,
        rank=rank,
        energy_threshold=energy_threshold,
        svd_threshold=svd_threshold,
    )
    return _dmd_impl(X, Y, resolved_rank)


def _validate_tensor_dmd_inputs(
    X: jax.Array,
    Y: jax.Array,
    L: LinearTransform,
    rank: int | None,
    energy_threshold: float | None,
    svd_threshold: float,
) -> None:
    if X.ndim != 3 or Y.ndim != 3:
        raise ValueError(
            f"tdmd expects third-order tensors X and Y; got ndim={X.ndim} and ndim={Y.ndim}."
        )
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape; got {X.shape} and {Y.shape}.")
    if min(X.shape) == 0:
        raise ValueError(f"X and Y must have positive dimensions; got shape {X.shape}.")
    if rank is not None:
        max_rank = min(X.shape[0], X.shape[1])
        if rank < 1 or rank > max_rank:
            raise ValueError(
                f"rank must be in [1, {max_rank}] for tensor shape {X.shape}; got {rank}."
            )
    _validate_truncation_policy(rank, energy_threshold, svd_threshold)
    if __debug__:
        L.debug_assert_last_axis_preserved(X)
        L.debug_assert_inverse_shape(X)


@partial(jax.jit, static_argnames=("rank",))
def _tdmd_impl(
    X: jax.Array,
    Y: jax.Array,
    L: LinearTransform,
    rank: int,
    svd_threshold: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    U, S, Vh = _truncated_tsvd_impl(X, L, rank)

    U_r = U
    S_r = S
    Vh_r = Vh

    U_r_h = L.t_transpose(U_r)
    V_r = L.t_transpose(Vh_r)
    S_r_inv = L.fdiag_pinv(S_r, svd_threshold)
    A_tilde = star_prod(star_prod(star_prod(U_r_h, Y, L), V_r, L), S_r_inv, L)
    W, D = L.eig_tensor(A_tilde)
    modes = star_prod(star_prod(star_prod(Y, V_r, L), S_r_inv, L), W, L)
    return modes, D


def tdmd(
    X: jax.Array,
    Y: jax.Array,
    L: LinearTransform,
    rank: int | None = None,
    *,
    energy_threshold: float | None = None,
    svd_threshold: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Compute tensor DMD modes and eigenvalues under the transform ``L``.

    Args:
        X: Snapshot tensor with shape ``(m, n, k)``.
        Y: Time-shifted snapshot tensor aligned with ``X``.
        L: Linear transform defining the tensor product algebra.
        rank: Optional truncation rank for the tensor SVD of ``X``.
        energy_threshold: Optional fraction of tube-spectrum energy to retain.
        svd_threshold: Absolute singular-value cutoff used in tensor pseudo-inversion.

    Returns:
        A tuple ``(modes, eigen_tensors)`` where ``modes`` contains the tensor
        DMD modes and ``eigen_tensors`` contains the diagonal eigen-tensor of
        the reduced operator.
    """
    _validate_tensor_dmd_inputs(X, Y, L, rank, energy_threshold, svd_threshold)
    spectrum = tensor_singular_spectrum(X, L)
    resolved_rank = _resolve_rank_from_spectrum(
        spectrum,
        rank=rank,
        energy_threshold=energy_threshold,
        svd_threshold=svd_threshold,
    )
    return _tdmd_impl(X, Y, L, resolved_rank, svd_threshold)
