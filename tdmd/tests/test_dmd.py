import jax
import jax.numpy as jnp
import pytest

from tdmd.core.dmd import tdmd, tdmdii
from tdmd.core.tensor_product import FFTTransform


def _random_tensor(key, shape):
    return jax.random.normal(key, shape)


def test_tdmd_shapes():
    key = jax.random.PRNGKey(10)
    X = _random_tensor(key, (5, 4, 3))
    Y = _random_tensor(key, (5, 4, 3))
    L = FFTTransform()

    modes, schur = tdmd(X, Y, L, svd_threshold=0.0)

    assert modes.shape == X.shape
    assert schur.shape[0] == schur.shape[1]


def test_tdmd_zero_input():
    X = jnp.zeros((4, 4, 2))
    Y = jnp.zeros((4, 4, 2))
    L = FFTTransform()

    modes, schur = tdmd(X, Y, L)

    assert jnp.allclose(modes, 0)
    assert jnp.allclose(schur, 0)


def test_tdmd_consistency():
    key = jax.random.PRNGKey(42)
    X = _random_tensor(key, (6, 5, 3))
    Y = X.copy()  # identity dynamics
    L = FFTTransform()

    modes, schur = tdmd(X, Y, L, svd_threshold=1e-6)

    norm_modes = jnp.linalg.norm(modes)
    schur_hat = L.to_slices(schur)
    eye = jnp.eye(schur_hat.shape[-1], dtype=schur_hat.dtype)[None, :, :]

    assert jnp.isfinite(norm_modes)
    assert norm_modes > 0
    assert jnp.allclose(schur_hat, eye, atol=1e-5)


def test_tdmd_signvals_threshold_zeros_small_modes():
    X = jnp.zeros((2, 2, 1), dtype=jnp.float32)
    X = X.at[0, 0, 0].set(2.0)
    X = X.at[1, 1, 0].set(0.25)
    Y = X
    L = FFTTransform()

    _, schur = tdmd(X, Y, L, signvals_threshold=0.5)
    schur_hat = L.to_slices(schur)

    assert jnp.allclose(jnp.diagonal(schur_hat, axis1=1, axis2=2), jnp.array([[1.0, 0.0]]))


def test_tdmd_rejects_invalid_signvals_threshold():
    X = jnp.ones((2, 2, 1))
    Y = X

    with pytest.raises(ValueError, match="finite non-negative threshold"):
        tdmd(X, Y, FFTTransform(), signvals_threshold=-1.0)


def test_tdmdii_shapes_and_multirank():
    X = jnp.zeros((3, 4, 2), dtype=jnp.float32)
    X = X.at[0, 0, 0].set(3.0)
    X = X.at[1, 1, 0].set(1.0)
    X = X.at[0, 0, 1].set(0.5)
    X = X.at[1, 1, 1].set(0.25)
    Y = X

    modes, schur, amplitudes, multirank = tdmdii(X, Y, FFTTransform(), gamma=0.95)

    assert modes.shape[0] == X.shape[0]
    assert modes.shape[2] == X.shape[2]
    assert schur.shape[0] == schur.shape[1]
    assert schur.shape[2] == X.shape[2]
    assert amplitudes.shape[1] == 1
    assert amplitudes.shape[2] == X.shape[2]
    assert multirank.shape == (X.shape[2],)
    assert int(multirank[0]) >= int(multirank[1])


def test_tdmdii_rejects_invalid_gamma():
    X = jnp.ones((2, 2, 1))
    Y = X

    with pytest.raises(ValueError, match="Expected gamma in"):
        tdmdii(X, Y, FFTTransform(), gamma=0.0)
