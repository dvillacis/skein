# Numerical regression fixtures

JSON fixtures with reference fits from the R packages `skein` is
positioned against (`glmnet`, `ncvreg`, `grpreg`). The Python
regression suite (`tests/test_r_regression.py`) loads these and
asserts skein's path solvers agree with the R reference along three
invariants: active-set match, sign agreement on meaningful
coefficients, and tight magnitude agreement at the smallest λ.

## Regenerating

R is **not** required to run pytest — fixtures are committed to the
repo, and missing-fixture tests skip cleanly. R is only needed to
*regenerate* the fixtures, which you do whenever:

- You bump a reference R package version and want to recheck.
- You add a new fixture for a new (penalty, family) combination.
- You suspect drift in skein and want a fresh comparison baseline.

### One-time R setup (macOS)

```bash
brew install r
# If install.packages fails with "fatal error: 'cmath' file not found",
# that's the R 4.6 / Apple Silicon SDK paths issue. Workaround:
mkdir -p ~/.R
cat > ~/.R/Makevars <<EOF
SDK = $(xcrun --sdk macosx --show-sdk-path)
CFLAGS  = -isysroot \$(SDK) -O2 -Wall
CXXFLAGS = -isysroot \$(SDK) -isystem \$(SDK)/usr/include/c++/v1 -O2 -Wall
CXX11FLAGS = -isysroot \$(SDK) -isystem \$(SDK)/usr/include/c++/v1 -O2 -Wall
CXX14FLAGS = -isysroot \$(SDK) -isystem \$(SDK)/usr/include/c++/v1 -O2 -Wall
CXX17FLAGS = -isysroot \$(SDK) -isystem \$(SDK)/usr/include/c++/v1 -O2 -Wall
CXX20FLAGS = -isysroot \$(SDK) -isystem \$(SDK)/usr/include/c++/v1 -O2 -Wall
EOF
```

### Install R packages

```r
install.packages(
    c("glmnet", "ncvreg", "grpreg", "survival", "jsonlite", "Rcpp", "RcppEigen"),
    repos = "https://cloud.r-project.org"
)
```

### Regenerate

From the repo root:

```bash
Rscript tests/fixtures/generate.R
```

This produces 8 JSON fixtures under `tests/fixtures/`. Verify with:

```bash
pytest tests/test_r_regression.py -v
```

Diff before committing — fixture content is bit-exact reproducible
given the seeds in `generate.R`, so a non-trivial diff means
something changed in the reference packages or the generator.

## What's in each fixture

| Fixture                            | R package   | Penalty | Family    | skein equivalent estimator                |
|------------------------------------|-------------|---------|-----------|-------------------------------------------|
| `glmnet_lasso_gaussian.json`       | glmnet 5.0  | lasso   | gaussian  | `MCPPathRegressor(gamma=1e6)`             |
| `ncvreg_mcp_gaussian.json`         | ncvreg 3.16 | MCP     | gaussian  | `MCPPathRegressor(gamma=3.0)`             |
| `ncvreg_scad_gaussian.json`        | ncvreg 3.16 | SCAD    | gaussian  | `SCADPathRegressor(a=3.7)`                |
| `grpreg_grlasso_gaussian.json`     | grpreg 3.6  | grLasso | gaussian  | `GroupLassoPathRegressor`                 |
| `grpreg_grmcp_gaussian.json`       | grpreg 3.6  | grMCP   | gaussian  | `GroupMCPPathRegressor(gamma=3.0)`        |
| `glmnet_lasso_binomial.json`       | glmnet 5.0  | lasso   | binomial  | `LogisticMCPPathRegressor(gamma=1e6)`     |
| `ncvreg_mcp_binomial.json`         | ncvreg 3.16 | MCP     | binomial  | `LogisticMCPPathRegressor(gamma=3.0)`     |
| `glmnet_lasso_cox.json`            | glmnet 5.0  | lasso   | cox       | `CoxMCPPathRegressor(gamma=1e6)`          |

Each JSON file contains:

- `X`, `y` (or `time` + `event` for Cox): the deterministic synthetic problem.
- `lambdas`: the λ-grid the reference solver chose.
- `coefs`: `(n_lambdas, n_features)` reference coefficients.
- `intercepts`: `(n_lambdas,)` reference intercepts (Cox excepted).
- `groups`, `group_multiplier`: present for group-penalty fixtures.
- Hyperparameters (`gamma`, `alpha`, `eps`, etc.) and metadata
  (`package`, `package_version`, `seed`).

## Why we don't run R in CI

CI builds need to be fast and free of system dependencies — adding
R + glmnet + ncvreg + grpreg + a Fortran toolchain to every PR is a
heavy ask. Committing the fixtures lets CI catch regressions
without R at PR time, and we regenerate locally when something
changes upstream.
