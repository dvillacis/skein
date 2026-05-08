# Installation

## From PyPI (recommended)

```bash
pip install skein-glm
```

Pre-built ABI3 wheels are published for:

- **Linux** x86_64 and aarch64
- **macOS** x86_64 (Intel) and arm64 (Apple Silicon)
- **Windows** AMD64

Wheels target Python 3.10+ (one ABI3 wheel per platform covers all
supported minor versions). Runtime dependencies are `numpy>=1.24`,
`scipy>=1.10`, and `scikit-learn>=1.3`.

## From source

If a wheel isn't available for your platform (or you want to develop
against the Rust core), build from source. You'll need a Rust
toolchain (stable channel) and `maturin`:

```bash
# 1. Install Rust if you don't have it.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Build + install the package into the active Python env.
pip install maturin
git clone https://github.com/dvillacis/skein
cd skein
maturin develop --release
```

`maturin develop` builds the Rust extension and registers an editable
install. Subsequent edits to the Rust core require re-running
`maturin develop --release`; edits to pure-Python code under
`python/skein/` take effect immediately.

## Development install

If you want to run the test suite, lint, or build docs locally:

```bash
maturin develop --release
pip install -e ".[dev,docs]"
```

The `[dev]` extras pull in `pytest`, `pytest-cov`, `ruff`, and `mypy`;
`[docs]` adds `mkdocs`, `mkdocs-material`, and `mkdocstrings`. After
that:

```bash
# Rust tests (fast iteration, no Python rebuild needed).
cargo test -p skein-core

# Python tests (require maturin develop first).
pytest

# Lint.
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
ruff check python/ tests/
mypy python/

# Docs preview at http://127.0.0.1:8000.
mkdocs serve
```

## Platform notes

### Apple Silicon (macOS arm64)

Wheels are native arm64 — no Rosetta translation. If you previously
ran `pip install` under x86_64 Python and switch to arm64 Python,
delete the cached wheel (`pip cache purge`) before reinstalling.

### Linux aarch64

The aarch64 wheel is built under QEMU emulation in CI (the test step
is skipped under emulation; we run the full suite on x86_64). On a
real aarch64 host the wheel runs at native speed.

### Windows

Pure MSVC build, no Cygwin / MSYS2 dependency. Building from source
requires the MSVC toolchain (Visual Studio Build Tools is enough);
`rustup` will pick it up automatically.

## Verifying the install

```python
import skein_glm
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((100, 10))
y = X[:, 0] - 2 * X[:, 1] + 0.1 * rng.standard_normal(100)
model = skein_glm.MCPRegressor(lambda_=0.05, gamma=3.0).fit(X, y)
print(model.coef_)         # should recover [~1, ~-2, 0, ..., 0]
print(model.intercept_)    # near 0
```

If that runs, you're good.
