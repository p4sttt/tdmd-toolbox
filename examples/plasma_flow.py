from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def make_plasma(nx: int = 100, ny: int = 100, nt: int = 100):
    x = jnp.linspace(-1.0, 1.0, nx)
    y = jnp.linspace(-1.0, 1.0, ny)
    t = jnp.linspace(0.0, 2.0 * jnp.pi, nt)
    X, Y, T = jnp.meshgrid(x, y, t, indexing="ij")

    Xf = 5.0 * X - 5.0
    Yf = 5.0 * Y - 5.0
    R = jnp.sqrt(Xf**2 + Yf**2)
    Theta = jnp.arctan2(Yf, Xf + 1.0e-6)

    Z = (
        7.0 * jnp.sin(1.7 * Xf + 0.9 * T)
        + 6.0 * jnp.cos(1.3 * Yf - 1.2 * T)
        + 5.0 * jnp.sin(0.9 * (Xf + Yf) + 0.7 * T)
        + 4.0 * jnp.cos(2.2 * R - 1.8 * T)
        + 3.0 * jnp.sin(3.0 * Theta + 0.6 * R - 1.1 * T)
        + 2.5 * jnp.cos(1.8 * (Xf - Yf) + 0.3 * R + 0.8 * T)
    )
    return Z, t


def split_shifted_tensor(sequence):
    return sequence[:, :, :-1], sequence[:, :, 1:]


def relative_error(target, approx):
    return float(jnp.linalg.norm(target - approx) / jnp.linalg.norm(target))


def draw_frame(ax, frame, title, cmap="magma"):
    image = ax.imshow(np.asarray(jnp.real(frame)), origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def apply_tdmd_operator(window, modes, schur_tensor, transform):
    modes_hat = transform.to_slices(modes)
    schur_hat = transform.to_slices(schur_tensor)
    window_hat = transform.to_slices(window)

    def apply_slice(phi, t, x):
        phi_pinv = jnp.linalg.pinv(phi)
        return phi @ t @ phi_pinv @ x

    next_hat = jax.vmap(apply_slice)(modes_hat, schur_hat, window_hat)
    return jnp.real(transform.from_slices(next_hat))
