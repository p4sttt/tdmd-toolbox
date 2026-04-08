from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsc

from tdmd.core.tensor_product import LinearTransform


def _validate_tensor_svd_inputs(A: jax.Array, L: LinearTransform) -> None:
    if A.ndim != 3:
        raise ValueError(f"Expected a third-order tensor with shape (m, n, k); got ndim={A.ndim}.")
    if min(A.shape) == 0:
        raise ValueError(f"Tensor dimensions must be positive; got shape {A.shape}.")
    if __debug__:
        L.debug_assert_last_axis_preserved(A)
        L.debug_assert_inverse_shape(A)


def _validate_truncation_policy(
    rank: int | None, energy_threshold: float | None, svd_threshold: float
) -> None:
    if rank is not None and rank < 1:
        raise ValueError(f"rank must be positive; got {rank}.")
    if energy_threshold is not None and not (0.0 < energy_threshold <= 1.0):
        raise ValueError(
            f"energy_threshold must be in the interval (0, 1]; got {energy_threshold}."
        )
    if svd_threshold < 0.0:
        raise ValueError(f"svd_threshold must be non-negative; got {svd_threshold}.")


def _resolve_rank_from_spectrum(
    spectrum: jax.Array,
    rank: int | None = None,
    energy_threshold: float | None = None,
    svd_threshold: float = 0.0,
) -> int:
    max_rank = int(spectrum.shape[0])
    resolved_rank = max_rank if rank is None else min(rank, max_rank)

    if energy_threshold is not None:
        energies = jnp.square(jnp.abs(spectrum))
        total_energy = float(jax.device_get(jnp.sum(energies)))
        if total_energy <= 0.0:
            raise ValueError(
                "Cannot resolve truncation rank because the singular spectrum has zero energy."
            )
        cumulative_energy = jax.device_get(jnp.cumsum(energies) / total_energy)
        energy_rank = int(jnp.searchsorted(cumulative_energy, energy_threshold, side="left")) + 1
        resolved_rank = min(resolved_rank, energy_rank)

    if svd_threshold > 0.0:
        tol_rank = int(jax.device_get(jnp.count_nonzero(jnp.abs(spectrum) > svd_threshold)))
        resolved_rank = min(resolved_rank, tol_rank)

    if resolved_rank < 1:
        raise ValueError(
            "Truncation policy removed all singular values; lower svd_threshold or energy_threshold."
        )
    return resolved_rank


@jax.jit
def _tsvd_slices_impl(A: jax.Array, L: LinearTransform) -> tuple[jax.Array, jax.Array, jax.Array]:
    A_hat = L.to_slices(A)
    return jsc.linalg.svd(A_hat, full_matrices=False)


@jax.jit
def _tsvd_impl(A: jax.Array, L: LinearTransform) -> tuple[jax.Array, jax.Array, jax.Array]:
    U_hat, Sigma, Vh_hat = _tsvd_slices_impl(A, L)
    rank = Sigma.shape[1]
    eye = jnp.eye(rank, dtype=Sigma.dtype)[None, :, :]
    S_hat = Sigma[:, :, None] * eye
    return L.from_slices(U_hat), L.from_slices(S_hat), L.from_slices(Vh_hat)


@partial(jax.jit, static_argnames=("rank",))
def _truncated_tsvd_impl(
    A: jax.Array, L: LinearTransform, rank: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    U, S, Vh = _tsvd_impl(A, L)
    return U[:, :rank, :], S[:rank, :rank, :], Vh[:rank, :, :]


def tsvd(A: jax.Array, L: LinearTransform) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the tensor SVD of ``A`` under the transform ``L``.

    Args:
        A: Input tensor with the transform axis in the last dimension.
        L: Linear transform defining the tensor product algebra.

    Returns:
        A tuple ``(U, S, Vh)`` such that ``A`` is factorized in the transform
        domain, with ``S`` storing the singular tubes as diagonal frontal slices.
    """
    _validate_tensor_svd_inputs(A, L)
    return _tsvd_impl(A, L)


def tensor_singular_spectrum(A: jax.Array, L: LinearTransform) -> jax.Array:
    """Return per-tube singular-value magnitudes used for truncation decisions."""
    _validate_tensor_svd_inputs(A, L)
    _, sing_vals, _ = _tsvd_slices_impl(A, L)
    return jnp.linalg.norm(sing_vals, axis=0)


def truncated_tsvd(
    A: jax.Array,
    L: LinearTransform,
    rank: int | None = None,
    *,
    energy_threshold: float | None = None,
    svd_threshold: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute a truncated tensor SVD of ``A`` under the transform ``L``."""
    _validate_tensor_svd_inputs(A, L)
    _validate_truncation_policy(rank, energy_threshold, svd_threshold)
    spectrum = tensor_singular_spectrum(A, L)
    resolved_rank = _resolve_rank_from_spectrum(
        spectrum,
        rank=rank,
        energy_threshold=energy_threshold,
        svd_threshold=svd_threshold,
    )
    return _truncated_tsvd_impl(A, L, resolved_rank)


@jax.jit
def _tschur_impl(A: jax.Array, L: LinearTransform) -> tuple[jax.Array, jax.Array]:
    A_hat = L.to_slices(A)
    T_hat, Z_hat = jsc.linalg.schur(A_hat)
    return L.from_slices(T_hat), L.from_slices(Z_hat)


def _validate_tensor_schur_inputs(A: jax.Array, L: LinearTransform) -> None:
    if A.ndim != 3:
        raise ValueError(f"Expected a third-order tensor with shape (m, m, k); got ndim={A.ndim}.")
    if min(A.shape) == 0:
        raise ValueError(f"Tensor dimensions must be positive; got shape {A.shape}.")
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Tensor Schur decomposition requires square frontal slices; got shape {A.shape}."
        )
    if __debug__:
        L.debug_assert_last_axis_preserved(A)
        L.debug_assert_inverse_shape(A)
        L.debug_assert_square_slices(A)


def tschur(A: jax.Array, L: LinearTransform) -> tuple[jax.Array, jax.Array]:
    """Compute the tensor Schur decomposition of ``A`` under the transform ``L``.

    Args:
        A: Input tensor with square frontal slices and transform axis last.
        L: Linear transform defining the tensor product algebra.

    Returns:
        A tuple ``(T, Z)`` where the transformed frontal slices satisfy the
        slice-wise Schur factorization ``A_hat = Z_hat @ T_hat @ Z_hat^H``.
    """
    _validate_tensor_schur_inputs(A, L)
    return _tschur_impl(A, L)
