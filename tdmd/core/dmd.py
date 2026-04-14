from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax.typing import ArrayLike

from tdmd.core.decomposition import (
    _truncate_tsvdii_impl,
    _truncated_tsvd_impl,
    _validate_gamma,
    _validate_tensor_input,
    _validate_threshold,
)
from tdmd.core.tensor_product import LinearTransform


class TDMDResult(NamedTuple):
    modes: jax.Array
    schur_tensor: jax.Array


class TDMDIIResult(NamedTuple):
    modes: jax.Array
    schur_tensor: jax.Array
    amplitudes: jax.Array
    multirank: jax.Array


def _resolve_check(check: bool) -> bool:
    if not isinstance(check, bool):
        raise TypeError(f"Expected check to be a bool; got {type(check).__name__}.")
    return check


def _validate_tensor_pair_inputs(
    X: ArrayLike, Y: ArrayLike, L: LinearTransform
) -> tuple[jax.Array, jax.Array]:
    X = _validate_tensor_input(X, L, require_square_slices=False)
    Y = _validate_tensor_input(Y, L, require_square_slices=False)
    if X.shape != Y.shape:
        raise ValueError(
            "Expected X and Y to have the same tensor shape; "
            f"got X.shape={X.shape} and Y.shape={Y.shape}."
        )
    return X, Y


def _invert_transformed_singular_values(S_hat: jax.Array, signvals_threshold: float) -> jax.Array:
    singular_values = jnp.diagonal(S_hat, axis1=1, axis2=2)
    keep = jnp.abs(singular_values) > signvals_threshold
    safe_singular_values = jnp.where(keep, singular_values, 1)
    singular_values_inv = jnp.where(keep, 1 / safe_singular_values, 0)

    eye = jnp.eye(S_hat.shape[1], dtype=S_hat.dtype)[None, :, :]
    return singular_values_inv[:, :, None] * eye


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


def _prepare_fit_tensors(
    X: ArrayLike,
    Y: ArrayLike | None,
    L: LinearTransform,
    *,
    check: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if Y is None:
        snapshots = (
            _validate_tensor_input(X, L, require_square_slices=False) if check else jnp.asarray(X)
        )
        if check and snapshots.shape[1] < 2:
            raise ValueError(
                "Expected at least two snapshots along axis 1 when Y is omitted; "
                f"got shape {snapshots.shape}."
            )
        return snapshots[:, :-1, :], snapshots[:, 1:, :], snapshots

    X_fit, Y_fit = (
        _validate_tensor_pair_inputs(X, Y, L) if check else (jnp.asarray(X), jnp.asarray(Y))
    )
    return X_fit, Y_fit, X_fit


def _initial_amplitudes(modes: jax.Array, X0: jax.Array, L: LinearTransform) -> jax.Array:
    modes_hat = L.to_slices(modes)
    X0_hat = L.to_slices(X0)

    def project_slice(phi, x0):
        return jnp.linalg.pinv(phi) @ x0

    amplitudes_hat = jax.vmap(project_slice)(modes_hat, X0_hat)
    return L.from_slices(amplitudes_hat)


def _forecast_tensor(
    modes: jax.Array,
    schur_tensor: jax.Array,
    amplitudes: jax.Array,
    horizon: int,
    L: LinearTransform,
) -> jax.Array:
    modes_hat = L.to_slices(modes)
    schur_hat = L.to_slices(schur_tensor)
    amplitudes_hat = L.to_slices(amplitudes)

    def predict_slice(phi, t, g, step):
        return phi @ jnp.linalg.matrix_power(t, step) @ g

    sequence_hat = []
    for step in range(horizon):
        sequence_hat.append(
            jax.vmap(predict_slice, in_axes=(0, 0, 0, None))(
                modes_hat,
                schur_hat,
                amplitudes_hat,
                step,
            )
        )
    return L.from_slices(jnp.concatenate(sequence_hat, axis=2))


def _predict_snapshot(
    modes: jax.Array,
    schur_tensor: jax.Array,
    amplitudes: jax.Array,
    step: int,
    L: LinearTransform,
) -> jax.Array:
    modes_hat = L.to_slices(modes)
    schur_hat = L.to_slices(schur_tensor)
    amplitudes_hat = L.to_slices(amplitudes)

    def predict_slice(phi, t, g):
        return phi @ jnp.linalg.matrix_power(t, step) @ g

    snapshot_hat = jax.vmap(predict_slice)(modes_hat, schur_hat, amplitudes_hat)
    return L.from_slices(snapshot_hat)[:, 0, :]


def _schur_eigenvalues(schur_tensor: jax.Array, L: LinearTransform) -> jax.Array:
    return jnp.diagonal(L.to_slices(schur_tensor), axis1=1, axis2=2)


class TDMD:
    """pyDMD-style wrapper for tensor DMD."""

    def __init__(
        self,
        transform: LinearTransform,
        *,
        svd_threshold: float = 0.0,
        signvals_threshold: float = 0.0,
        check: bool = True,
    ) -> None:
        self.transform = transform
        self.svd_threshold = svd_threshold
        self.signvals_threshold = signvals_threshold
        self.check = _resolve_check(check)
        self._reset()

    def _reset(self) -> None:
        self._snapshots: jax.Array | None = None
        self._modes: jax.Array | None = None
        self._schur_tensor: jax.Array | None = None
        self._amplitudes: jax.Array | None = None
        self._reconstructed_data: jax.Array | None = None

    def _require_fit(self) -> None:
        if self._modes is None or self._schur_tensor is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before use.")

    def fit(self, X: ArrayLike, Y: ArrayLike | None = None, *, check: bool | None = None) -> TDMD:
        check = self.check if check is None else _resolve_check(check)
        X_fit, Y_fit, snapshots = _prepare_fit_tensors(X, Y, self.transform, check=check)
        modes, schur_tensor = _fit_tdmd(
            X_fit,
            Y_fit,
            self.transform,
            svd_threshold=self.svd_threshold,
            signvals_threshold=self.signvals_threshold,
            check=check,
        )
        amplitudes = _initial_amplitudes(modes, snapshots[:, :1, :], self.transform)

        self._snapshots = snapshots
        self._modes = modes
        self._schur_tensor = schur_tensor
        self._amplitudes = amplitudes
        self._reconstructed_data = _forecast_tensor(
            modes,
            schur_tensor,
            amplitudes,
            snapshots.shape[1],
            self.transform,
        )
        return self

    def predict_next(self) -> jax.Array:
        self._require_fit()
        return self.predict_step(1)

    def predict_step(self, step: int) -> jax.Array:
        self._require_fit()
        step = int(step)
        if step < 0:
            raise ValueError(f"Expected a non-negative step; got {step}.")
        return _predict_snapshot(
            self.modes,
            self.schur_tensor,
            self.amplitudes,
            step,
            self.transform,
        )

    def predict(self) -> jax.Array:
        return self.predict_next()

    def forecast(self, horizon: int) -> jax.Array:
        self._require_fit()
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"Expected a positive forecast horizon; got {horizon}.")
        return _forecast_tensor(
            self.modes,
            self.schur_tensor,
            self.amplitudes,
            horizon,
            self.transform,
        )

    @property
    def snapshots(self) -> jax.Array:
        self._require_fit()
        return self._snapshots

    @property
    def modes(self) -> jax.Array:
        self._require_fit()
        return self._modes

    @property
    def schur_tensor(self) -> jax.Array:
        self._require_fit()
        return self._schur_tensor

    @property
    def amplitudes(self) -> jax.Array:
        self._require_fit()
        return self._amplitudes

    @property
    def eigs(self) -> jax.Array:
        self._require_fit()
        return _schur_eigenvalues(self.schur_tensor, self.transform)

    @property
    def reconstructed_data(self) -> jax.Array:
        self._require_fit()
        return self._reconstructed_data


class TDMDII:
    """pyDMD-style wrapper for the multirank TDMDII variant."""

    def __init__(
        self,
        transform: LinearTransform,
        *,
        gamma: float = 0.99999,
        signvals_threshold: float = 0.0,
        check: bool = True,
    ) -> None:
        self.transform = transform
        self.gamma = gamma
        self.signvals_threshold = signvals_threshold
        self.check = _resolve_check(check)
        self._reset()

    def _reset(self) -> None:
        self._snapshots: jax.Array | None = None
        self._modes: jax.Array | None = None
        self._schur_tensor: jax.Array | None = None
        self._amplitudes: jax.Array | None = None
        self._multirank: jax.Array | None = None
        self._reconstructed_data: jax.Array | None = None

    def _require_fit(self) -> None:
        if self._modes is None or self._schur_tensor is None or self._amplitudes is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before use.")

    def fit(self, X: ArrayLike, Y: ArrayLike | None = None, *, check: bool | None = None) -> TDMDII:
        check = self.check if check is None else _resolve_check(check)
        X_fit, Y_fit, snapshots = _prepare_fit_tensors(X, Y, self.transform, check=check)
        modes, schur_tensor, amplitudes, multirank = _fit_tdmdii(
            X_fit,
            Y_fit,
            self.transform,
            gamma=self.gamma,
            signvals_threshold=self.signvals_threshold,
            check=check,
        )

        self._snapshots = snapshots
        self._modes = modes
        self._schur_tensor = schur_tensor
        self._amplitudes = amplitudes
        self._multirank = multirank
        self._reconstructed_data = _forecast_tensor(
            modes,
            schur_tensor,
            amplitudes,
            snapshots.shape[1],
            self.transform,
        )
        return self

    def predict_next(self) -> jax.Array:
        self._require_fit()
        return self.predict_step(1)

    def predict_step(self, step: int) -> jax.Array:
        self._require_fit()
        step = int(step)
        if step < 0:
            raise ValueError(f"Expected a non-negative step; got {step}.")
        return _predict_snapshot(
            self.modes,
            self.schur_tensor,
            self.amplitudes,
            step,
            self.transform,
        )

    def predict(self) -> jax.Array:
        return self.predict_next()

    def forecast(self, horizon: int) -> jax.Array:
        self._require_fit()
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"Expected a positive forecast horizon; got {horizon}.")
        return _forecast_tensor(
            self.modes,
            self.schur_tensor,
            self.amplitudes,
            horizon,
            self.transform,
        )

    @property
    def snapshots(self) -> jax.Array:
        self._require_fit()
        return self._snapshots

    @property
    def modes(self) -> jax.Array:
        self._require_fit()
        return self._modes

    @property
    def schur_tensor(self) -> jax.Array:
        self._require_fit()
        return self._schur_tensor

    @property
    def amplitudes(self) -> jax.Array:
        self._require_fit()
        return self._amplitudes

    @property
    def multirank(self) -> jax.Array:
        self._require_fit()
        return self._multirank

    @property
    def eigs(self) -> jax.Array:
        self._require_fit()
        return _schur_eigenvalues(self.schur_tensor, self.transform)

    @property
    def reconstructed_data(self) -> jax.Array:
        self._require_fit()
        return self._reconstructed_data


@partial(jax.jit, static_argnames=("svd_threshold", "signvals_threshold"))
def _tdmd_impl(
    X: ArrayLike,
    Y: ArrayLike,
    L: LinearTransform,
    svd_threshold: float,
    signvals_threshold: float,
) -> TDMDResult:
    X_hat = L.to_slices(X)
    U_hat, S_hat, Vh_hat = _truncated_tsvd_impl(X_hat, L, svd_threshold)
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
    amplitudes_hat = jax.vmap(lambda phi, x0: jnp.linalg.pinv(phi) @ x0)(modes_hat, X0_hat)

    return TDMDIIResult(
        L.from_slices(modes_hat),
        L.from_slices(T_hat),
        L.from_slices(amplitudes_hat),
        multirank,
    )


def _fit_tdmd(
    X: ArrayLike,
    Y: ArrayLike,
    L: LinearTransform,
    svd_threshold: float = 0.0,
    signvals_threshold: float = 0.0,
    *,
    check: bool = True,
) -> TDMDResult:
    check = _resolve_check(check)
    X, Y = _validate_tensor_pair_inputs(X, Y, L) if check else (jnp.asarray(X), jnp.asarray(Y))
    svd_threshold = _validate_threshold(svd_threshold) if check else float(svd_threshold)
    signvals_threshold = (
        _validate_threshold(signvals_threshold) if check else float(signvals_threshold)
    )
    return _tdmd_impl(X, Y, L, svd_threshold, signvals_threshold)


def _fit_tdmdii(
    X: ArrayLike,
    Y: ArrayLike,
    L: LinearTransform,
    gamma: float = 0.99999,
    signvals_threshold: float = 0.0,
    *,
    check: bool = True,
) -> TDMDIIResult:
    check = _resolve_check(check)
    X, Y = _validate_tensor_pair_inputs(X, Y, L) if check else (jnp.asarray(X), jnp.asarray(Y))
    gamma = _validate_gamma(gamma) if check else float(gamma)
    signvals_threshold = (
        _validate_threshold(signvals_threshold) if check else float(signvals_threshold)
    )
    return _tdmdii_impl(X, Y, L, gamma, signvals_threshold)
