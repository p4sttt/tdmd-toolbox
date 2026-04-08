# Tensor Dynamic Mode Decomposition Toolbox

Python JAX implementation of Tensor Dynamic Mode Decomposition Toolbox. It contains implementation of tensor algebra with invertible linear transformation and some decomposition algorithms based on it.

## Installation

First clone git repository

```bash
git clone https://github.com/p4sttt/tdmd-toolbox
```

Create virtual environment and install dependencies using [uv](https://docs.astral.sh/uv/) or over dependency manager what u prefer

```bash
uv venv
uv sync
```

## What those package contains

They have the following functionality

- Invertible transform abstractions for tensor algebra, including `MatrixTransform` and `FFTTransform`
- Tensor t-product primitives via `LinearTransform` and the `star_prod` helper
- Tensor decompositions `tsvd` and `truncated_tsvd` for factorizing third-order tensors
- Dynamic Mode Decomposition for matrices with `dmd`
- Tensor Dynamic Mode Decomposition with `tdmd` under a chosen transform basis

## Examples

<!-- TODO: describe numerical experiments in /examples -->
