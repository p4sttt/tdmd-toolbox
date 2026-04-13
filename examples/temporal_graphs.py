from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import gzip

import jax.numpy as jnp
import numpy as np
import scipy.linalg as la
from scipy.fft import dct

from tdmd import MatrixTransform, tdmdii


SNAP_TEMPORAL_GRAPH_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal-Dept3.txt.gz"


@dataclass(frozen=True)
class TemporalGraphData:
    snapshots: np.ndarray
    node_ids: np.ndarray
    bin_edges: np.ndarray


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
        urlretrieve(SNAP_TEMPORAL_GRAPH_URL, path)
    return path


def relative_error(target: np.ndarray, approx: np.ndarray) -> float:
    denom = np.linalg.norm(target)
    return float(np.linalg.norm(target - approx) / (denom if denom > 1.0e-12 else 1.0))


def _load_temporal_edges(path: Path) -> np.ndarray:
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            src, dst, ts = map(int, line.split())
            rows.append((src, dst, ts))
    return np.asarray(rows, dtype=np.int64)


def load_temporal_graph(path: Path, *, num_bins: int = 32) -> TemporalGraphData:
    path = ensure_dataset(path)
    rows = _load_temporal_edges(path)

    node_ids = np.unique(rows[:, :2])
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    rows[:, 0] = np.vectorize(node_to_idx.get)(rows[:, 0])
    rows[:, 1] = np.vectorize(node_to_idx.get)(rows[:, 1])

    timestamps = np.sort(rows[:, 2])
    bin_edges = np.quantile(timestamps, np.linspace(0.0, 1.0, num_bins + 1))
    bin_edges = np.floor(bin_edges).astype(np.int64)
    bin_edges[0] = timestamps.min()
    bin_edges[-1] = timestamps.max() + 1
    for idx in range(1, len(bin_edges)):
        if bin_edges[idx] <= bin_edges[idx - 1]:
            bin_edges[idx] = bin_edges[idx - 1] + 1

    num_nodes = len(node_ids)
    snapshots = np.zeros((num_nodes, num_bins, num_nodes), dtype=np.float64)
    for src, dst, ts in rows:
        bin_idx = min(np.searchsorted(bin_edges, ts, side="right") - 1, num_bins - 1)
        snapshots[src, bin_idx, dst] += 1.0

    return TemporalGraphData(snapshots=snapshots, node_ids=node_ids, bin_edges=bin_edges)


def statewise_relative_errors(states: np.ndarray, reconstructed_states: np.ndarray) -> np.ndarray:
    return np.array(
        [
            relative_error(states[:, idx], reconstructed_states[:, idx])
            for idx in range(states.shape[1])
        ]
    )


def flatten_tensor_states(snapshots: np.ndarray) -> np.ndarray:
    return snapshots.transpose(0, 2, 1).reshape(
        snapshots.shape[0] * snapshots.shape[2], snapshots.shape[1], order="F"
    )


def reconstruct_dmd(snapshots: np.ndarray, rank: int) -> DMDResult:
    states = flatten_tensor_states(snapshots)
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


def reconstruct_tdmdii(snapshots: np.ndarray, gamma: float) -> TDMDIIResult:
    transform = MatrixTransform(jnp.asarray(dct_matrix(snapshots.shape[2])))

    X = jnp.asarray(snapshots[:, :-1, :])
    Y = jnp.asarray(snapshots[:, 1:, :])
    modes, schur_tensor, amplitudes, multirank = tdmdii(X, Y, transform, gamma=gamma)

    modes_hat = np.asarray(transform.to_slices(modes))
    schur_hat = np.asarray(transform.to_slices(schur_tensor))
    amplitudes_hat = np.asarray(transform.to_slices(amplitudes))
    multirank = np.asarray(multirank)

    reconstructed_hat = np.zeros(
        (snapshots.shape[2], snapshots.shape[0], snapshots.shape[1]), dtype=np.complex128
    )
    for step in range(snapshots.shape[1]):
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
    states = flatten_tensor_states(snapshots)
    reconstructed_states = flatten_tensor_states(reconstructed_tensor)
    state_errors = statewise_relative_errors(states, reconstructed_states)
    storage_cost = int(
        snapshots.shape[0] * multirank.sum()
        + sum(int(k_face * (k_face + 1) // 2 + k_face) for k_face in multirank)
    )
    return TDMDIIResult(
        reconstructed_tensor,
        reconstructed_states,
        state_errors,
        storage_cost,
        multirank,
    )


def display_graph_snapshot(snapshot: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(snapshot))
