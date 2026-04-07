from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp


class LinearTransform(ABC):
    """Abstract interface for invertible linear transforms."""

    @abstractmethod
    def apply(self, x: jax.Array) -> jax.Array:
        """Apply the transform"""

    @abstractmethod
    def apply_inverse(self, x: jax.Array) -> jax.Array:
        """Apply the inverse transform"""

    def to_slices(self, x: jax.Array) -> jax.Array:
        """Map a tensor to transformed frontal slices with shape ``(k, m, n)``."""
        return self.apply(x).transpose((2, 0, 1))

    def from_slices(self, x_hat: jax.Array) -> jax.Array:
        """Map transformed frontal slices with shape ``(k, m, n)`` back to tensor form."""
        return self.apply_inverse(x_hat.transpose((1, 2, 0)))

    def star_prod(self, A: jax.Array, B: jax.Array) -> jax.Array:
        """Compute the t-product induced by this transform."""
        return self.from_slices(self.to_slices(A) @ self.to_slices(B))

    def t_transpose(self, A: jax.Array) -> jax.Array:
        """Tensor transpose defined by conjugate-transposing transformed slices."""
        A_hat_h = jnp.swapaxes(jnp.conjugate(self.to_slices(A)), 1, 2)
        return self.from_slices(A_hat_h)

    def fdiag_inverse(self, A: jax.Array) -> jax.Array:
        """Invert a tensor whose transformed frontal slices are diagonal matrices."""
        A_hat = self.to_slices(A)
        diag = jnp.diagonal(A_hat, axis1=1, axis2=2)
        eye = jnp.eye(A.shape[0], dtype=A_hat.dtype)[None, :, :]
        return self.from_slices((1.0 / diag)[:, :, None] * eye)

    def fdiag_pinv(self, A: jax.Array, threshold: float = 0.0) -> jax.Array:
        """Pseudo-invert diagonal transformed slices, zeroing entries below ``threshold``."""
        A_hat = self.to_slices(A)
        diag = jnp.diagonal(A_hat, axis1=1, axis2=2)
        mask = jnp.abs(diag) > threshold
        safe_diag = jnp.where(mask, diag, jnp.ones_like(diag))
        inv_diag = jnp.where(mask, 1.0 / safe_diag, jnp.zeros_like(diag))
        eye = jnp.eye(A.shape[0], dtype=A_hat.dtype)[None, :, :]
        return self.from_slices(inv_diag[:, :, None] * eye)

    def eig_tensor(self, A: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Eigen-decompose each transformed frontal slice and map back."""
        A_hat = self.to_slices(A)
        eigenvalues, W_hat = jnp.linalg.eig(A_hat)
        D_hat = eigenvalues[:, :, None] * jnp.eye(A.shape[0], dtype=eigenvalues.dtype)[None, :, :]
        return self.from_slices(W_hat), self.from_slices(D_hat)

    def debug_assert_last_axis_preserved(self, x: jax.Array) -> None:
        """Assert that the transform preserves the input tensor shape."""
        x_hat = self.apply(x)
        assert x_hat.shape == x.shape, "Transform must preserve tensor shape."

    def debug_assert_inverse_shape(self, x: jax.Array) -> None:
        """Assert that the inverse transform preserves the input tensor shape."""
        x_roundtrip = self.apply_inverse(self.apply(x))
        assert x_roundtrip.shape == x.shape, "Inverse transform must preserve tensor shape."

    def debug_assert_square_slices(self, A: jax.Array) -> None:
        """Assert that transformed frontal slices are square matrices."""
        assert A.ndim == 3, "Tensor operations require a third-order tensor."
        assert A.shape[0] == A.shape[1], "Operation requires square frontal slices."

    def debug_assert_star_prod_args(self, A: jax.Array, B: jax.Array) -> None:
        """Assert shape compatibility for the tensor product."""
        assert A.ndim == 3 and B.ndim == 3, "Tensor product requires third-order tensors."
        assert A.shape[1] == B.shape[0], "Inner tensor-product dimensions must match."
        assert A.shape[2] == B.shape[2], "Transform-axis sizes must match."


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class MatrixTransform(LinearTransform):
    """Linear transform defined by an invertible matrix on the last axis."""

    M: jax.Array
    M_T: jax.Array
    condition_threshold: float | None
    condition_number: float

    def __init__(self, M: jax.Array, *, condition_threshold: float | None = 1.0e8) -> None:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"MatrixTransform expects a square matrix; got shape {M.shape}.")

        cond = float(jax.device_get(jnp.linalg.cond(M)))
        if not jnp.isfinite(cond):
            raise ValueError("MatrixTransform requires an invertible finite matrix.")
        if condition_threshold is not None and cond > condition_threshold:
            warnings.warn(
                (
                    "MatrixTransform matrix is ill-conditioned "
                    f"(cond={cond:.3e} > {condition_threshold:.3e}); "
                    "inverse applications may be numerically unstable."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        object.__setattr__(self, "M", M)
        object.__setattr__(self, "M_T", M.T)
        object.__setattr__(self, "condition_threshold", condition_threshold)
        object.__setattr__(self, "condition_number", cond)

    def apply(self, x: jax.Array) -> jax.Array:
        return jnp.tensordot(x, self.M_T, ([2], [0]))

    def apply_inverse(self, x: jax.Array) -> jax.Array:
        x_shape = x.shape
        x_flat = x.reshape((-1, x_shape[-1]))
        solved = jnp.linalg.solve(self.M, x_flat.T).T
        return solved.reshape(x_shape)

    def tree_flatten(self) -> tuple[tuple[jax.Array], tuple[float | None, float]]:
        return (self.M,), (self.condition_threshold, self.condition_number)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[float | None, float], children: tuple[jax.Array]
    ) -> MatrixTransform:
        condition_threshold, condition_number = aux_data
        obj = cls.__new__(cls)
        (M,) = children
        object.__setattr__(obj, "M", M)
        object.__setattr__(obj, "M_T", M.T)
        object.__setattr__(obj, "condition_threshold", condition_threshold)
        object.__setattr__(obj, "condition_number", condition_number)
        return obj


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FFTTransform(LinearTransform):
    """FFT-based transform on the last axis."""

    def apply(self, x: jax.Array) -> jax.Array:
        return jnp.fft.fft(x, axis=-1)

    def apply_inverse(self, x: jax.Array) -> jax.Array:
        return jnp.fft.ifft(x, axis=-1)

    def tree_flatten(self) -> tuple[tuple[()], None]:
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[()]) -> FFTTransform:
        del aux_data, children
        return cls()


def star_prod(A: jax.Array, B: jax.Array, L: LinearTransform) -> jax.Array:
    """Compute the t-product of third-order tensors under a chosen basis.

    Args:
        A: Left tensor with shape ``(m, n, k)``.
        B: Right tensor with shape ``(n, p, k)``.
        L: Invertible transform applied along the third axis before slice-wise
            matrix multiplication and inverted afterward.

    Returns:
        The tensor t-product of ``A`` and ``B`` with shape ``(m, p, k)``.

    Notes:
        The transform object owns the algebraic operations in its transform
        domain, so future transform types can customize behavior by overriding
        ``LinearTransform`` methods.
    """
    return L.star_prod(A, B)
