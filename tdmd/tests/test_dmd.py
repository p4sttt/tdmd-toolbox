import jax
import jax.numpy as jnp
import pytest

from tdmd.core.dmd import TDMD, TDMDII
from tdmd.core.tensor_product import FFTTransform


def _random_tensor(key, shape):
    return jax.random.normal(key, shape)


def test_tdmd_fit_shapes():
    key = jax.random.PRNGKey(10)
    snapshots = _random_tensor(key, (5, 6, 3))
    model = TDMD(FFTTransform(), svd_threshold=0.0)

    model.fit(snapshots)

    assert model.modes.shape == snapshots[:, :-1, :].shape
    assert model.schur_tensor.shape[0] == model.schur_tensor.shape[1]
    assert model.reconstructed_data.shape == snapshots.shape


def test_tdmd_zero_input_reconstruction():
    snapshots = jnp.zeros((4, 5, 2))
    model = TDMD(FFTTransform())

    model.fit(snapshots)

    assert jnp.allclose(model.modes, 0)
    assert jnp.allclose(model.schur_tensor, 0)
    assert jnp.allclose(model.reconstructed_data, 0)


def test_tdmd_consistency_for_identity_dynamics():
    key = jax.random.PRNGKey(42)
    X = _random_tensor(key, (6, 5, 3))
    model = TDMD(FFTTransform(), svd_threshold=1e-6)

    model.fit(X, X)

    schur_hat = model.transform.to_slices(model.schur_tensor)
    eye = jnp.eye(schur_hat.shape[-1], dtype=schur_hat.dtype)[None, :, :]

    assert jnp.isfinite(jnp.linalg.norm(model.modes))
    assert jnp.allclose(schur_hat, eye, atol=1e-5)


def test_tdmd_rejects_invalid_signvals_threshold():
    snapshots = jnp.ones((2, 3, 1))
    model = TDMD(FFTTransform(), signvals_threshold=-1.0)

    with pytest.raises(ValueError, match="finite non-negative threshold"):
        model.fit(snapshots[:, :-1, :], snapshots[:, 1:, :])


def test_tdmd_can_skip_threshold_validation():
    snapshots = jnp.ones((2, 3, 1))
    model = TDMD(FFTTransform(), signvals_threshold=-1.0, check=False)

    model.fit(snapshots[:, :-1, :], snapshots[:, 1:, :], check=False)

    assert model.modes.shape == snapshots[:, :-1, :].shape
    assert model.schur_tensor.shape[2] == snapshots.shape[2]


def test_tdmdii_shapes_and_multirank():
    snapshots = jnp.zeros((3, 5, 2), dtype=jnp.float32)
    snapshots = snapshots.at[0, :, 0].set(jnp.array([3.0, 2.0, 1.0, 0.5, 0.25]))
    snapshots = snapshots.at[1, :, 1].set(jnp.array([1.0, 0.75, 0.5, 0.25, 0.125]))
    model = TDMDII(FFTTransform(), gamma=0.95)

    model.fit(snapshots)

    assert model.modes.shape[0] == snapshots.shape[0]
    assert model.modes.shape[2] == snapshots.shape[2]
    assert model.schur_tensor.shape[0] == model.schur_tensor.shape[1]
    assert model.schur_tensor.shape[2] == snapshots.shape[2]
    assert model.amplitudes.shape[1] == 1
    assert model.amplitudes.shape[2] == snapshots.shape[2]
    assert model.multirank.shape == (snapshots.shape[2],)
    assert int(model.multirank[0]) >= int(model.multirank[1])


def test_tdmdii_rejects_invalid_gamma():
    snapshots = jnp.ones((2, 3, 1))
    model = TDMDII(FFTTransform(), gamma=0.0)

    with pytest.raises(ValueError, match="Expected a finite gamma in"):
        model.fit(snapshots[:, :-1, :], snapshots[:, 1:, :])


def test_tdmd_wrapper_prediction_api():
    key = jax.random.PRNGKey(7)
    snapshots = _random_tensor(key, (5, 6, 3))
    model = TDMD(FFTTransform(), svd_threshold=1.0e-6)

    model.fit(snapshots)

    assert model.predict_next().shape == (snapshots.shape[0], snapshots.shape[2])
    assert model.predict_step(4).shape == (snapshots.shape[0], snapshots.shape[2])
    assert model.predict().shape == (snapshots.shape[0], snapshots.shape[2])
    assert model.forecast(7).shape == (snapshots.shape[0], 7, snapshots.shape[2])


def test_tdmdii_wrapper_prediction_api():
    snapshots = jnp.zeros((3, 5, 2), dtype=jnp.float32)
    snapshots = snapshots.at[0, :, 0].set(jnp.array([3.0, 2.0, 1.0, 0.5, 0.25]))
    snapshots = snapshots.at[1, :, 1].set(jnp.array([1.0, 0.75, 0.5, 0.25, 0.125]))
    model = TDMDII(FFTTransform(), gamma=0.95)

    model.fit(snapshots)

    assert model.predict_next().shape == (snapshots.shape[0], snapshots.shape[2])
    assert model.predict_step(3).shape == (snapshots.shape[0], snapshots.shape[2])
    assert model.forecast(6).shape == (snapshots.shape[0], 6, snapshots.shape[2])


def test_tdmd_wrapper_rejects_negative_step():
    model = TDMD(FFTTransform())
    model.fit(jnp.ones((2, 3, 1)))

    with pytest.raises(ValueError, match="non-negative step"):
        model.predict_step(-1)
