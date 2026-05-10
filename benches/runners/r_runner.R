# R runner for the M9 bench suite.
#
# Invoked from Python via `Rscript benches/runners/r_runner.R <input.json>
# <output.json>`. The Python caller writes the problem (X, y, lambdas,
# package, penalty, family) to <input.json>; we fit and write the result
# (coef_path, fit_time_s, package, version, active_set_size) to
# <output.json>.
#
# Why this shape: keeps R deps out of the Python venv, lets the Python
# driver treat R as just-another-runner, and ensures both languages see
# byte-identical X / y.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript r_runner.R <input.json> <output.json>")
}
input_path <- args[[1]]
output_path <- args[[2]]

input <- fromJSON(input_path, simplifyMatrix = TRUE)
package <- input$package
penalty <- input$penalty
family <- if (is.null(input$family)) "gaussian" else input$family
X <- as.matrix(input$X)
y <- as.numeric(input$y)
lambdas <- as.numeric(input$lambdas)
tol <- if (is.null(input$tol)) 1e-6 else as.numeric(input$tol)
groups <- if (is.null(input$groups)) NULL else as.integer(input$groups)

fit_result <- list()

if (package == "glmnet") {
  suppressPackageStartupMessages(library(glmnet))
  glmnet_family <- switch(
    family,
    gaussian = "gaussian",
    logistic = "binomial",
    poisson = "poisson",
    stop(sprintf("glmnet: unsupported family %s", family))
  )
  alpha <- if (penalty == "lasso") 1.0 else if (penalty == "elastic_net") 0.5 else stop("glmnet: bad penalty")
  t0 <- proc.time()
  fit <- glmnet(X, y, family = glmnet_family, alpha = alpha, lambda = lambdas, thresh = tol, standardize = FALSE)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  coef_path <- as.matrix(fit$beta)  # p × n_lambdas
  fit_result <- list(
    package = "glmnet",
    version = as.character(packageVersion("glmnet")),
    fit_time_s = elapsed,
    coef_path = t(coef_path),
    active_set_size = sum(coef_path[, ncol(coef_path)] != 0)
  )
} else if (package == "ncvreg") {
  suppressPackageStartupMessages(library(ncvreg))
  ncv_penalty <- switch(
    penalty,
    mcp = "MCP",
    scad = "SCAD",
    lasso = "lasso",
    stop(sprintf("ncvreg: unsupported penalty %s", penalty))
  )
  ncv_family <- switch(
    family,
    gaussian = "gaussian",
    logistic = "binomial",
    poisson = "poisson",
    stop(sprintf("ncvreg: unsupported family %s", family))
  )
  t0 <- proc.time()
  fit <- ncvreg(X, y, family = ncv_family, penalty = ncv_penalty, lambda = lambdas, eps = tol)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  coef_path <- as.matrix(fit$beta)[-1, , drop = FALSE]  # drop intercept row, p × n_lambdas
  fit_result <- list(
    package = "ncvreg",
    version = as.character(packageVersion("ncvreg")),
    fit_time_s = elapsed,
    coef_path = t(coef_path),
    active_set_size = sum(coef_path[, ncol(coef_path)] != 0)
  )
} else if (package == "grpreg") {
  suppressPackageStartupMessages(library(grpreg))
  if (is.null(groups)) stop("grpreg: groups required")
  grp_penalty <- switch(
    penalty,
    group_lasso = "grLasso",
    group_mcp = "grMCP",
    group_scad = "grSCAD",
    stop(sprintf("grpreg: unsupported penalty %s", penalty))
  )
  grp_family <- switch(
    family,
    gaussian = "gaussian",
    logistic = "binomial",
    poisson = "poisson",
    stop(sprintf("grpreg: unsupported family %s", family))
  )
  t0 <- proc.time()
  fit <- grpreg(X, y, group = groups, family = grp_family, penalty = grp_penalty, lambda = lambdas, eps = tol)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  coef_path <- as.matrix(fit$beta)[-1, , drop = FALSE]
  fit_result <- list(
    package = "grpreg",
    version = as.character(packageVersion("grpreg")),
    fit_time_s = elapsed,
    coef_path = t(coef_path),
    active_set_size = sum(coef_path[, ncol(coef_path)] != 0)
  )
} else {
  stop(sprintf("unknown R package: %s", package))
}

write(toJSON(fit_result, auto_unbox = TRUE, digits = 8), output_path)
