from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.typing import ArrayLike

from tdmd.core.tensor_product import LinearTransform


class TSVDResult(NamedTuple):
    U: jax.Array
    S: jax.Array
    Vh: jax.Array


class TSVDIIResult(NamedTuple):
    U: jax.Array
    S: jax.Array
    Vh: jax.Array
    multirank: jax.Array


class TSchurResult(NamedTuple):
    T: jax.Array
    W: jax.Array


def _validate_tensor_input(
    A: ArrayLike, L: LinearTransform, *, require_square_slices: bool
) -> jax.Array:
    A = jnp.asarray(A)
    if A.ndim != 3:
        raise ValueError(
            f"Expected a third-order tensor with shape (m, n, k); got shape {A.shape}."
        )

    transform_size = getattr(L, "M", None)
    if transform_size is not None and transform_size.shape[0] != A.shape[2]:
        raise ValueError(
            "Input tensor third-axis length must match the transform size; "
            f"got tensor shape {A.shape} and transform size {transform_size.shape[0]}."
        )

    if require_square_slices and A.shape[0] != A.shape[1]:
        raise ValueError(
            "Tensor Schur decomposition requires square frontal slices; " f"got shape {A.shape}."
        )

    return A


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if threshold < 0 or not jnp.isfinite(threshold):
        raise ValueError(f"Expected a finite non-negative threshold; got {threshold}.")
    return threshold


def _validate_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if gamma <= 0 or gamma > 1 or not jnp.isfinite(gamma):
        raise ValueError(f"Expected a finite gamma in the range [0, 1]; got {gamma}.")
    return gamma


@jax.jit
def _tsvd_impl(A_hat: ArrayLike, L: LinearTransform) -> TSVDResult:
    # U_hat: (k, m, rmax), S_hat: (k, rmax), Vh_hat: (k, rmax, n), where rmax = min(m, n)
    U_hat, S_hat, Vh_hat = jnp.linalg.svd(A_hat, full_matrices=False)

    rank = S_hat.shape[1]
    eye = jnp.eye(rank, dtype=S_hat.dtype)[None, :, :]
    S_hat = S_hat[:, :, None] * eye

    return TSVDResult(U_hat, S_hat, Vh_hat)


@partial(jax.jit, static_argnames=("threshold", "singvals_threshold"))
def _truncated_tsvd_impl(
    A_hat: ArrayLike,
    L: LinearTransform,
    threshold: float,
    singvals_threshold: float = 0.0,
) -> TSVDResult:
    # U_hat: (k, m, rmax), S_hat: (k, rmax), Vh_hat: (k, rmax, n), where rmax = min(m, n)
    U_hat, S_hat, Vh_hat = jnp.linalg.svd(A_hat, full_matrices=False)

    tube_spectrum = jnp.linalg.norm(S_hat, axis=0)  # (rmax,)
    keep_tubes = tube_spectrum > threshold
    keep_singular_values = S_hat > singvals_threshold

    S_hat = jnp.where(keep_tubes[None, :], S_hat, 0.0)
    U_hat = jnp.where(keep_tubes[None, None, :], U_hat, 0.0)
    Vh_hat = jnp.where(keep_tubes[None, :, None], Vh_hat, 0.0)
    S_hat = jnp.where(keep_singular_values, S_hat, 0.0)

    rmax = S_hat.shape[1]
    eye = jnp.eye(rmax, dtype=S_hat.dtype)[None, :, :]
    S_hat = S_hat[:, :, None] * eye

    return TSVDResult(U_hat, S_hat, Vh_hat)


@partial(jax.jit, static_argnames=("gamma",))
def _truncate_tsvdii_impl(A_hat: jax.Array, gamma: float) -> TSVDIIResult:
    U_hat, S_hat, Vh_hat = jnp.linalg.svd(A_hat, full_matrices=False)

    energies = (jnp.abs(S_hat) ** 2).reshape(-1)
    sorted_energies = jnp.sort(energies)[::-1]
    cumulative_energy = jnp.cumsum(sorted_energies)
    # keep largests k sungular values such that
    # sum_{i=1}^k sigma_i^2 >= gamma * sum_{i=1}^r sigma_i^2
    keep_count = jnp.searchsorted(cumulative_energy, gamma * cumulative_energy[-1], side="left") + 1
    cutoff = sorted_energies[keep_count - 1]

    keep = (jnp.abs(S_hat) ** 2) >= cutoff
    masked_singular_values = jnp.where(keep, S_hat, 0)
    multirank = jnp.sum(keep, axis=1)

    return TSVDIIResult(
        U_hat,
        masked_singular_values,
        Vh_hat,
        multirank,
    )


@jax.jit
def _tschur_impl(A_hat: ArrayLike, L: LinearTransform) -> TSchurResult:
    T_hat, W_hat = jsp.linalg.schur(A_hat)
    return TSchurResult(T_hat, W_hat)


def tsvd(A: jax.Array, L: LinearTransform, *, check: bool = True) -> TSVDResult:
    """Compute the tensor SVD of a third-order tensor under transform ``L``.

    Args:
        A: Tensor with shape ``(m, n, k)``.
        L: Linear transform defining the tensor product algebra along axis 2.

    Returns:
        A tuple ``(U, S, Vh)`` in tensor form such that ``A = U * S * Vh``
        under the t-product induced by ``L``.

    Raises:
        ValueError: If ``A`` is not third-order or is incompatible with ``L``.
    """
    A = _validate_tensor_input(A, L, require_square_slices=False) if check else jnp.asarray(A)

    A_hat = L.to_slices(A)
    U_hat, S_hat, Vh_hat = _tsvd_impl(A_hat, L)

    return TSVDResult(
        L.from_slices(U_hat),
        L.from_slices(S_hat),
        L.from_slices(Vh_hat),
    )


def truncated_tsvd(
    A: jax.Array,
    L: LinearTransform,
    threshold: float = 0,
    singvals_threshold: float = 0,
    *,
    check: bool = True,
) -> TSVDResult:
    """Compute a thresholded tensor SVD under transform ``L``.

    Args:
        A: Tensor with shape ``(m, n, k)``.
        L: Linear transform defining the tensor product algebra along axis 2.
        threshold: Absolute cutoff applied to the tube spectrum. Components
            whose tube norm is not greater than ``threshold`` are discarded.
        singvals_threshold: Absolute cutoff applied to each transformed
            singular value after tube truncation. Singular values not greater
            than this threshold are set to zero slice-wise.

    Returns:
        A tuple ``(U, S, Vh)`` with discarded tensor singular components set
        to zero in the transform domain.

    Raises:
        ValueError: If ``A`` is not third-order, is incompatible with ``L``,
            or if a threshold is negative or non-finite.
    """
    A = _validate_tensor_input(A, L, require_square_slices=False) if check else jnp.asarray(A)
    threshold = _validate_threshold(threshold) if check else float(threshold)
    singvals_threshold = (
        _validate_threshold(singvals_threshold) if check else float(singvals_threshold)
    )

    A_hat = L.to_slices(A)
    U_hat, S_hat, Vh_hat = _truncated_tsvd_impl(A_hat, L, threshold, singvals_threshold)

    return TSVDResult(
        L.from_slices(U_hat),
        L.from_slices(S_hat),
        L.from_slices(Vh_hat),
    )


def truncated_tsvdii(
    A: jax.Array, L: LinearTransform, gamma: float, *, check: bool = True
) -> TSVDIIResult:
    """Compute the truncated tensor SVD under transform ``L``.

    Args:
        A: Tensor with shape ``(m, n, k)``
        L: Linear transform defining the tensor product algebra along axis 2.
        gamma: Truncation parameter.

    Returns:
        A tuple ``(U, S, Vh)`` with discarded tensor singular components set
        to zero in the transform domain.
    """
    A = _validate_tensor_input(A, L, require_square_slices=False) if check else jnp.asarray(A)
    gamma = _validate_gamma(gamma) if check else float(gamma)

    A_hat = L.to_slices(A)
    U_hat, S_hat, Vh_hat, multirank = _truncate_tsvdii_impl(A_hat, gamma)

    return TSVDIIResult(
        L.from_slices(U_hat),
        L.from_slices(S_hat),
        L.from_slices(Vh_hat),
        multirank,
    )


def tschur(A: ArrayLike, L: LinearTransform, *, check: bool = True) -> TSchurResult:
    """Compute the tensor Schur decomposition under transform ``L``.

    Args:
        A: Tensor with shape ``(m, m, k)``.
        L: Linear transform defining the tensor product algebra along axis 2.

    Returns:
        A tuple ``(T, W)`` in tensor form, where ``T`` is upper triangular in
        the transform domain and ``W`` contains the corresponding unitary basis.

    Raises:
        ValueError: If ``A`` is not third-order, has non-square frontal slices,
            or is incompatible with ``L``.
    """
    A = _validate_tensor_input(A, L, require_square_slices=True) if check else jnp.asarray(A)

    A_hat = L.to_slices(A)
    T_hat, W_hat = _tschur_impl(A_hat, L)

    return TSchurResult(
        L.from_slices(T_hat),
        L.from_slices(W_hat),
    )
