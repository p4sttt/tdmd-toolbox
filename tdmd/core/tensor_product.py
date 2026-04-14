from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.typing import ArrayLike


class LinearTransform(ABC):
    """Abstract interface for invertible linear transforms."""

    @abstractmethod
    def apply(self, x: ArrayLike) -> jax.Array:
        raise NotImplementedError

    @abstractmethod
    def apply_inverse(self, x: ArrayLike) -> jax.Array:
        raise NotImplementedError

    def to_slices(self, x: ArrayLike) -> jax.Array:
        """Map a tensor to transformed frontal slices with shape ``(k, m, n)``."""
        return jnp.transpose(self.apply(x), (2, 0, 1))

    def from_slices(self, x_hat: ArrayLike) -> jax.Array:
        """Map transformed frontal slices with shape ``(k, m, n)`` back to tensor form."""
        return self.apply_inverse(jnp.transpose(x_hat, (1, 2, 0)))

    def star_prod(self, A: ArrayLike, B: ArrayLike) -> jax.Array:
        """Compute the t-product induced by this transform."""
        return self.from_slices(self.to_slices(A) @ self.to_slices(B))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MatrixTransform(LinearTransform):
    """Linear transform defined by an invertible matrix on the last axis."""

    M: jax.Array

    @classmethod
    def from_matrix(cls, M: ArrayLike, check: bool = True) -> MatrixTransform:
        M = jnp.asarray(M)

        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Expected a square matrix; got shape {M.shape}.")

        if check:
            cond = jnp.linalg.cond(M)
            if not jnp.isfinite(cond):
                raise ValueError("Matrix must be invertible.")

        return cls(M)

    def apply(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.tensordot(x, self.M.T, ([2], [0]))

    def apply_inverse(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        x_shape = x.shape
        x_flat = jnp.reshape(x, (-1, x_shape[-1]))
        solved = jnp.linalg.solve(self.M, x_flat.T).T
        return solved.reshape(x_shape)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FFTTransform(LinearTransform):
    """FFT-based transform on the last axis."""

    def apply(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.fft.fft(x, axis=-1)

    def apply_inverse(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return jnp.fft.ifft(x, axis=-1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DCTTransform(LinearTransform):
    """DCT-based transform on the last axis."""

    def apply(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return jsp.fft.dct(x, axis=-1, norm="ortho")

    def apply_inverse(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return jsp.fft.idct(x, axis=-1, norm="ortho")


def star_prod(A: ArrayLike, B: ArrayLike, L: LinearTransform) -> jax.Array:
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
