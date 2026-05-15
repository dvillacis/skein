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

# Optional — only needed for the M14a R-anchor fixtures
# (`psych_polychoric.json` and `glmnet_cox_active_set.json`).
# Mainstream R has no Cox debiased lasso, so the Cox anchor compares
# against `glmnet(family='cox')` active sets; the polychoric anchor
# uses `psych::polychoric()`. Both blocks in `generate.R` skip
# cleanly with a message if the package is missing.
install.packages(c("psych"), repos = "https://cloud.r-project.org")
```

### Regenerate

From the repo root:

```bash
Rscript tests/fixtures/generate.R
```

This produces 8 small-tier JSON fixtures (n=200–300, p=15–24) plus 3
mid-tier fixtures at n=500, p=100 (`*_mid.json`). Verify with:

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
| `glmnet_lasso_gaussian_mid.json`   | glmnet 5.0  | lasso   | gaussian  | `MCPPathRegressor(gamma=1e6)`  (n=500, p=100) |
| `ncvreg_mcp_gaussian_mid.json`     | ncvreg 3.16 | MCP     | gaussian  | `MCPPathRegressor(gamma=3.0)`  (n=500, p=100) |
| `glmnet_lasso_binomial_mid.json`   | glmnet 5.0  | lasso   | binomial  | `LogisticMCPPathRegressor(gamma=1e6)` (n=500, p=100) |
| `psych_polychoric.json`            | psych 2.x   | (correlation) | ordinal | `polychoric_correlation()` (M14a R-anchor; n=500, p=8) |
| `glmnet_cox_active_set.json`       | glmnet 5.0  | lasso   | cox       | `debiased_cox_lasso()` active set (M14a R-anchor; n=400, p=25) |

The mid-tier (`*_mid.json`) is an M14c.3 addition that exercises the
path solvers at a size where LLA local-min divergence on nonconvex
problems matters and the active-set fuzz grows with p. Tolerances on
the Python side (`tests/test_r_regression.py`) are looser for the
mid-tier: `smallest_lambda_atol` 5e-3–5e-2 vs 1e-5 on the small
tier, and `active_set_fuzz_frac` 0.15 vs the default 0.10. A
regression that only fires at scale (e.g. a screening rule applied
to the wrong λ index, fixed-cost outer overhead that scales
super-linearly) gets caught here when the small tier would miss it.

The M14a R-anchor fixtures (`psych_polychoric.json` and
`glmnet_cox_active_set.json`) gate the v0.9 M14a deliverables
against independent R references. The polychoric anchor is a
tight elementwise comparison (atol=5e-3) — Olsson two-step ML is
well-conditioned, both implementations should land on the MLE. The
Cox anchor is **not** a debiased reference (mainstream R has none
for Cox); it's a Jaccard ≥ 0.6 active-set agreement vs
`glmnet(family='cox')` — useful as a regression gate against
variable-selection bugs, weaker than the polychoric anchor.

Both are designed to be committed to the repo once regenerated
(small enough — `psych_polychoric.json` ≈ 250 KB,
`glmnet_cox_active_set.json` ≈ 200 KB). They use their own inline
`pytest.skip()` on missing fixtures (not the strict
`_skipped_if_missing` helper), so they soft-skip in CI even under
`SKEIN_REQUIRE_FIXTURES=1` until a maintainer regenerates and
commits them. After commit, the tests run everywhere.

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
