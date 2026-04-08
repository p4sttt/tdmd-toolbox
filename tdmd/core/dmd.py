from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsc

from tdmd.core.decomposition import (
    _resolve_rank_from_spectrum,
    _tsvd_slices_impl,
    _validate_truncation_policy,
)
from tdmd.core.tensor_product import LinearTransform


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
    U_hat: jax.Array,
    S_sigma: jax.Array,
    Vh_hat: jax.Array,
    Y: jax.Array,
    L: LinearTransform,
    rank: int,
    svd_threshold: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    Y_hat = L.to_slices(Y)
    U_hat = U_hat[:, :, :rank]
    singular_vals = S_sigma[:, :rank]
    Vh_hat = Vh_hat[:, :rank, :]

    U_h_hat = jnp.swapaxes(U_hat.conj(), 1, 2)
    V_hat = jnp.swapaxes(Vh_hat.conj(), 1, 2)

    singular_vals_inv = jnp.where(
        jnp.abs(singular_vals) > svd_threshold,
        1.0 / singular_vals,
        0.0,
    )
    eye = jnp.eye(rank, dtype=singular_vals.dtype)[None, :, :]
    S_inv_hat = singular_vals_inv[:, :, None] * eye

    A_tilde_hat = U_h_hat @ Y_hat @ V_hat @ S_inv_hat
    T_hat, W_hat = jsc.linalg.schur(A_tilde_hat)
    modes_hat = U_hat @ W_hat
    return L.from_slices(modes_hat), L.from_slices(T_hat)


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
        A tuple ``(modes, schur_tensor)`` where ``modes`` contains the tensor
        DMD modes and ``schur_tensor`` contains the tensor Schur form of the
        reduced operator.
    """
    _validate_tensor_dmd_inputs(X, Y, L, rank, energy_threshold, svd_threshold)
    U_hat, sigma, Vh_hat = _tsvd_slices_impl(X, L)
    spectrum = jnp.linalg.norm(sigma, axis=0)
    resolved_rank = _resolve_rank_from_spectrum(
        spectrum,
        rank=rank,
        energy_threshold=energy_threshold,
        svd_threshold=svd_threshold,
    )
    return _tdmd_impl(U_hat, sigma, Vh_hat, Y, L, resolved_rank, svd_threshold)
