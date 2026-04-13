from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import jax.numpy as jnp
import numpy as np
import scipy.io as sio
import scipy.linalg as la
from scipy.fft import dct

from tdmd import MatrixTransform, tdmdii


CYLINDER_DATA_URL = (
    "https://raw.githubusercontent.com/dynamicslab/databook_python/master/DATA/VORTALL.mat"
)
CYLINDER_GRID_SHAPE = (199, 449)
CYLINDER_SNAPSHOT_COUNT = 150


@dataclass(frozen=True)
class DMDResult:
    reconstructed_states: np.ndarray
    state_errors: np.ndarray
    storage_cost: int


@dataclass(frozen=True)
class TDMDIIResult:
    reconstructed_tensor: np.ndarray
    reconstructed_states: np.ndarray
    state_errors: np.ndarray
    storage_cost: int
    multirank: np.ndarray


def ensure_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(CYLINDER_DATA_URL, path)
    return path


def load_cylinder_flow(
    path: Path,
    *,
    snapshot_count: int = CYLINDER_SNAPSHOT_COUNT,
    grid_shape: tuple[int, int] = CYLINDER_GRID_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    path = ensure_dataset(path)
    states = sio.loadmat(path)["VORTALL"][:, :snapshot_count]
    nx, ny = grid_shape
    tensor = states.reshape(nx, ny, snapshot_count, order="F").transpose(0, 2, 1)
    return states, tensor


def reshape_snapshot(
    vector: np.ndarray, grid_shape: tuple[int, int] = CYLINDER_GRID_SHAPE
) -> np.ndarray:
    nx, ny = grid_shape
    return vector.reshape(nx, ny, order="F")


def relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(target - approx) / np.linalg.norm(target))


def statewise_relative_errors(states: np.ndarray, reconstructed_states: np.ndarray) -> np.ndarray:
    return np.array(
        [relative_error(states[:, idx], reconstructed_states[:, idx]) for idx in range(states.shape[1])]
    )


def flatten_tensor_states(tensor: np.ndarray) -> np.ndarray:
    return tensor.transpose(0, 2, 1).reshape(tensor.shape[0] * tensor.shape[2], tensor.shape[1], order="F")


def reconstruct_dmd(states: np.ndarray, rank: int) -> DMDResult:
    X = states[:, :-1]
    Y = states[:, 1:]

    U, singular_values, Vh = la.svd(X, full_matrices=False)
    U = U[:, :rank]
    singular_values = singular_values[:rank]
    Vh = Vh[:rank, :]

    K = U.conj().T @ Y @ Vh.conj().T @ np.diag(1.0 / singular_values)
    T, W = la.schur(K, output="complex")
    modes = U @ W
    amplitudes = modes.conj().T @ states[:, [0]]

    reconstructed_states = np.empty_like(states, dtype=np.complex128)
    for step in range(states.shape[1]):
        reconstructed_states[:, [step]] = modes @ np.linalg.matrix_power(T, step) @ amplitudes

    reconstructed_states = reconstructed_states.real
    state_errors = statewise_relative_errors(states, reconstructed_states)
    storage_cost = states.shape[0] * rank + rank * (rank + 1) // 2
    return DMDResult(reconstructed_states, state_errors, storage_cost)


def dct_matrix(size: int) -> np.ndarray:
    return dct(np.eye(size), type=2, norm="ortho", axis=0)


def reconstruct_tdmdii(tensor: np.ndarray, gamma: float) -> TDMDIIResult:
    transform = MatrixTransform(jnp.asarray(dct_matrix(tensor.shape[2])))

    X = jnp.asarray(tensor[:, :-1, :])
    Y = jnp.asarray(tensor[:, 1:, :])
    modes, schur_tensor, amplitudes, multirank = tdmdii(X, Y, transform, gamma=gamma)

    modes_hat = np.asarray(transform.to_slices(modes))
    schur_hat = np.asarray(transform.to_slices(schur_tensor))
    amplitudes_hat = np.asarray(transform.to_slices(amplitudes))
    multirank = np.asarray(multirank)

    reconstructed_hat = np.zeros((tensor.shape[2], tensor.shape[0], tensor.shape[1]), dtype=np.complex128)
    for step in range(tensor.shape[1]):
        for face, k_face in enumerate(multirank):
            k_face = int(k_face)
            if k_face == 0:
                continue
            reconstructed_hat[face, :, step : step + 1] = (
                modes_hat[face, :, :k_face]
                @ np.linalg.matrix_power(schur_hat[face, :k_face, :k_face], step)
                @ amplitudes_hat[face, :k_face, :]
            )

    reconstructed_tensor = np.asarray(transform.from_slices(jnp.asarray(reconstructed_hat))).real
    reconstructed_states = flatten_tensor_states(reconstructed_tensor)
    original_states = flatten_tensor_states(tensor)

    state_errors = statewise_relative_errors(original_states, reconstructed_states)
    storage_cost = int(
        tensor.shape[0] * multirank.sum()
        + sum(int(k_face * (k_face + 1) // 2 + k_face) for k_face in multirank)
    )
    return TDMDIIResult(
        reconstructed_tensor,
        reconstructed_states,
        state_errors,
        storage_cost,
        multirank,
    )
