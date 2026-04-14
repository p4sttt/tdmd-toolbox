import jax
import jax.numpy as jnp
import pytest

from tdmd.core.decomposition import tsvd, truncated_tsvd, tschur
from tdmd.core.tensor_product import FFTTransform, MatrixTransform


def _random_tensor(key, shape):
    return jax.random.normal(key, shape)


def test_tsvd_reconstruction():
    key = jax.random.PRNGKey(0)
    A = _random_tensor(key, (5, 4, 3))
    L = FFTTransform()

    U, S, Vh = tsvd(A, L)

    A_reconstructed = L.star_prod(L.star_prod(U, S), Vh)

    assert jnp.allclose(A, A_reconstructed, atol=1e-5)


def test_tsvd_shapes():
    key = jax.random.PRNGKey(1)
    A = _random_tensor(key, (6, 5, 4))
    L = FFTTransform()

    U, S, Vh = tsvd(A, L)

    assert U.shape == (6, 5, 4)
    assert S.shape == (5, 5, 4)
    assert Vh.shape == (5, 5, 4)


def test_truncated_tsvd_threshold_effect():
    key = jax.random.PRNGKey(2)
    A = _random_tensor(key, (5, 5, 3))
    L = FFTTransform()

    _, S_full, _ = truncated_tsvd(A, L, threshold=0.0)
    _, S_trunc, _ = truncated_tsvd(A, L, threshold=1.0)

    norm_full = jnp.linalg.norm(S_full)
    norm_trunc = jnp.linalg.norm(S_trunc)

    assert norm_trunc <= norm_full


def test_truncated_tsvd_zero_tensor():
    A = jnp.zeros((4, 4, 2))
    L = FFTTransform()

    U, S, Vh = truncated_tsvd(A, L, threshold=1e-6)

    assert jnp.allclose(S, 0)
    assert jnp.allclose(U, 0)
    assert jnp.allclose(Vh, 0)


def test_truncated_tsvd_singular_value_threshold_effect():
    A = jnp.zeros((2, 2, 1), dtype=jnp.float32)
    A = A.at[0, 0, 0].set(2.0)
    A = A.at[1, 1, 0].set(0.25)
    L = FFTTransform()

    _, S, _ = truncated_tsvd(A, L, singvals_threshold=0.5)
    singular_values = jnp.diagonal(L.to_slices(S), axis1=1, axis2=2)

    assert jnp.allclose(singular_values, jnp.array([[2.0, 0.0]], dtype=singular_values.dtype))


def test_tschur_reconstruction():
    key = jax.random.PRNGKey(3)
    A = _random_tensor(key, (4, 4, 3))
    L = FFTTransform()

    T, Z = tschur(A, L)

    A_hat = L.to_slices(A)
    T_hat = L.to_slices(T)
    Z_hat = L.to_slices(Z)

    reconstructed = Z_hat @ T_hat @ jnp.swapaxes(Z_hat.conj(), 1, 2)

    assert jnp.allclose(A_hat, reconstructed, atol=1e-5)


def test_tsvd_rejects_non_third_order_input():
    A = jnp.ones((4, 4))

    with pytest.raises(ValueError, match="third-order tensor"):
        tsvd(A, FFTTransform())


def test_truncated_tsvd_rejects_invalid_threshold():
    A = jnp.ones((4, 4, 2))

    with pytest.raises(ValueError, match="finite non-negative threshold"):
        truncated_tsvd(A, FFTTransform(), threshold=-1.0)


def test_truncated_tsvd_rejects_invalid_singular_value_threshold():
    A = jnp.ones((4, 4, 2))

    with pytest.raises(ValueError, match="finite non-negative threshold"):
        truncated_tsvd(A, FFTTransform(), singvals_threshold=-1.0)


def test_truncated_tsvd_can_skip_threshold_validation():
    A = jnp.ones((4, 4, 2))
    U, S, Vh = truncated_tsvd(
        A,
        FFTTransform(),
        threshold=-1.0,
        singvals_threshold=-1.0,
        check=False,
    )

    assert U.shape == (4, 4, 2)
    assert S.shape == (4, 4, 2)
    assert Vh.shape == (4, 4, 2)


def test_tsvd_rejects_transform_axis_mismatch():
    A = jnp.ones((4, 4, 3))
    L = MatrixTransform(jnp.eye(2))

    with pytest.raises(ValueError, match="third-axis length must match the transform size"):
        tsvd(A, L)


def test_tschur_requires_square_slices():
    A = jnp.ones((4, 3, 2))

    with pytest.raises(ValueError, match="requires square frontal slices"):
        tschur(A, FFTTransform())
