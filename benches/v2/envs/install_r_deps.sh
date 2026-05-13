#!/usr/bin/env bash
# Install the R packages required by the v2 R-backed comparators.
#
# Heads up: `arrow` is a large CRAN package (compiles the Apache Arrow
# C++ libraries on macOS/Linux without prebuilt binaries) — expect
# 5-10 min on first install. The other packages are fast.
#
# Usage:
#   bash benches/v2/envs/install_r_deps.sh

set -euo pipefail

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript not on PATH — install R first (e.g. via Homebrew: 'brew install r')." >&2
  exit 1
fi

REPO=${R_REPO:-https://cloud.r-project.org/}

Rscript -e "
  pkgs <- c('arrow', 'glmnet', 'ncvreg', 'grpreg', 'glasso')
  missing <- pkgs[!pkgs %in% installed.packages()[, 1]]
  if (length(missing)) {
    message('Installing: ', paste(missing, collapse=', '))
    install.packages(missing, repos='${REPO}')
  } else {
    message('All v2 R deps already installed.')
  }
  for (p in pkgs) {
    cat(p, '==', as.character(packageVersion(p)), '\n')
  }
"
