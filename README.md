# Tensor Dynamic Mode Decomposition Toolbox

`tdmd-toolbox` is a JAX-based library for tensor dynamic mode decomposition
under transform-induced tensor algebras.

The package currently focuses on:

- tensor DMD (`tdmd`)
- multirank TDMDII (`tdmdii`) inspired by the `star_M`-product formulation
- configurable transform bases through `FFTTransform` and `MatrixTransform`

Classical matrix DMD is not part of the library API. It is used only inside the
example notebooks for side-by-side comparisons against tensor methods.

## Installation

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/p4sttt/tdmd-toolbox
cd tdmd-toolbox
uv venv
uv sync
```

The project requires Python `>=3.12`.

## Library API

Public exports from `tdmd`:

- `FFTTransform`
- `MatrixTransform`
- `tdmd`
- `tdmdii`

### Transforms

- `FFTTransform` applies an FFT along the third tensor axis.
- `MatrixTransform` uses an explicit invertible square matrix on the third axis.

Both transforms define the tensor algebra used by the TDMD algorithms.

### Internal algebra layer

The repository also contains low-level algebra and decomposition primitives in
`tdmd.core`, including tensor products, tensor SVD variants, and tensor Schur
decomposition. They are kept as internal or advanced building blocks rather
than top-level library entry points.

### Tensor DMD

- `tdmd(X, Y, L, svd_threshold=..., signvals_threshold=...)` computes tensor DMD
  modes and a tensor Schur operator for the reduced dynamics.
- `tdmdii(X, Y, L, gamma=..., signvals_threshold=...)` computes the multirank
  TDMDII variant and additionally returns initial amplitudes and the retained
  face-wise multirank.

`tdmdii` is the article-style multirank variant used in the cylinder and graph
examples.

## Quick Start

```python
import jax.numpy as jnp

from tdmd import FFTTransform, tdmd

L = FFTTransform()
X = jnp.ones((8, 15, 4))
Y = X

modes, schur_tensor = tdmd(
    X,
    Y,
    L,
    svd_threshold=0.0,
    signvals_threshold=1.0e-8,
)
```

For the multirank TDMDII variant:

```python
import jax.numpy as jnp

from tdmd import MatrixTransform, tdmdii

M = jnp.eye(6)
L = MatrixTransform(M)

X = jnp.ones((10, 20, 6))
Y = X

modes, schur_tensor, amplitudes, multirank = tdmdii(
    X,
    Y,
    L,
    gamma=0.999,
    signvals_threshold=1.0e-8,
)
```

## Examples

The `examples/` directory contains notebooks and small helper modules for three
comparison settings.

### Plasma

- `examples/dmd-plasma.ipynb`
- `examples/tdmd-plasma.ipynb`
- `examples/plasma_flow.py`

These notebooks compare classical matrix DMD against `tdmdii` on a synthetic
plasma-like space-time field. Both notebooks evaluate reconstruction error over
multiple forecasted snapshots and plot error growth across the prediction
horizon.

### Cylinder Flow

- `examples/dmd-cylinder.ipynb`
- `examples/tdmd-cylinder.ipynb`
- `examples/cylinder_flow.py`

These notebooks compare matrix DMD and `tdmdii` on the cylinder wake dataset
from the Dynamics Lab data repository. The TDMDII setup uses a DCT-based
`MatrixTransform` and reproduces the article-style comparison where tensor
structure is beneficial.

### Temporal Graphs

- `examples/dmd-temporal-graph.ipynb`
- `examples/tdmd-temporal-graph.ipynb`
- `examples/temporal_graphs.py`

These notebooks compare matrix DMD and `tdmdii` on temporal graph snapshots
constructed from the Stanford SNAP `email-Eu-core-temporal-Dept3` dataset.

### Example data downloads

Some notebooks download data automatically on first run:

- cylinder wake data from the Dynamics Lab public dataset
- temporal graph data from Stanford SNAP

The downloads are cached under `examples/data/`.

## Testing

Run the test suite with:

```bash
./.venv/bin/pytest tdmd/tests
```

At the time of this README update, the repository test suite passes with:

- `18 passed`

## Project Structure

```text
tdmd/
  core/
    tensor_product.py
    decomposition.py
    dmd.py
  tests/
examples/
  *.ipynb
  *.py
```

## References

- [Third-Order Tensors as Linear Operators on a Space of Matrices](https://www.sciencedirect.com/science/article/pii/S0024379510002934)
- [Tensor-tensor products with invertible linear transforms](https://www.sciencedirect.com/science/article/pii/S0024379515004358)
- [Tensor Dynamic Mode Decomposition](https://arxiv.org/abs/2508.02627)
- [A tensor-based dynamic mode decomposition based on the star-M-product](https://arxiv.org/abs/2508.10126)
