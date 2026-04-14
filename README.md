# Tensor Dynamic Mode Decomposition Toolbox

`tdmd-toolbox` is a JAX-based library for tensor dynamic mode decomposition
under transform-induced tensor algebras.

The package currently focuses on:

- tensor DMD (`TDMD`)
- multirank TDMDII (`TDMDII`) inspired by the `star_M`-product formulation
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
- `TDMD`
- `TDMDII`

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

- `TDMD(transform, ...)` provides a model-style `fit`, `predict_next()`,
  `predict_step(step)`, and `forecast(horizon)` interface for tensor DMD.
- `TDMDII(transform, ...)` provides the same model-style API for the multirank
  TDMDII variant and additionally exposes `multirank`.

`TDMDII` is the article-style multirank variant used in the cylinder and graph
examples.

## Quick Start

```python
import jax.numpy as jnp

from tdmd import FFTTransform, TDMD

L = FFTTransform()
snapshots = jnp.ones((8, 16, 4))

model = TDMD(
    L,
    svd_threshold=0.0,
    signvals_threshold=1.0e-8,
)
model.fit(snapshots)

modes = model.modes
schur_tensor = model.schur_tensor
next_snapshot = model.predict_next()
step_5_snapshot = model.predict_step(5)
```

For the multirank TDMDII variant:

```python
import jax.numpy as jnp

from tdmd import MatrixTransform, TDMDII

M = jnp.eye(6)
L = MatrixTransform(M)

snapshots = jnp.ones((10, 21, 6))

model = TDMDII(
    L,
    gamma=0.999,
    signvals_threshold=1.0e-8,
)
model.fit(snapshots)

modes = model.modes
schur_tensor = model.schur_tensor
amplitudes = model.amplitudes
multirank = model.multirank
next_snapshot = model.predict_next()
```

## Examples

The `examples/` directory contains notebooks for three comparison settings.

### Plasma

- `examples/dmd-plasma.ipynb`
- `examples/tdmd-plasma.ipynb`

These notebooks compare classical matrix DMD against `TDMDII` on a synthetic
plasma-like space-time field. Both notebooks evaluate reconstruction error over
multiple forecasted snapshots and plot error growth across the prediction
horizon.

### Cylinder Flow

- `examples/dmd-cylinder.ipynb`
- `examples/tdmd-cylinder.ipynb`

These notebooks compare matrix DMD and `TDMDII` on the cylinder wake dataset
from the Dynamics Lab data repository. The TDMDII setup uses a DCT-based
transform and reproduces the article-style comparison where tensor structure is
beneficial.

### Temporal Graphs

- `examples/dmd-temporal-graph.ipynb`
- `examples/tdmd-temporal-graph.ipynb`

These notebooks compare matrix DMD and `TDMDII` on temporal graph snapshots
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

- `22 passed`

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
