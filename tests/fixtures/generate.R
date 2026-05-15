# Generate reference fits from glmnet / ncvreg / grpreg on canonical
# synthetic problems. Run from the repo root:
#
#     Rscript tests/fixtures/generate.R
#
# This produces JSON fixtures under tests/fixtures/ that the
# Python regression suite (tests/test_r_regression.py) loads and
# compares against the equivalent skein fits.
#
# We use small problems (n=200, p=20-30) so:
#   1. JSON encoding of X / coefs is manageable.
#   2. Both R and skein converge essentially exactly, which lets
#      us assert ≤ 1e-5 agreement instead of having to budget for
#      different convergence behavior on harder problems.
#
# All fits use deterministic seeds; rerunning should be a no-op
# diff modulo the FP noise of the reference solver.

suppressPackageStartupMessages({
  library(glmnet)
  library(ncvreg)
  library(grpreg)
  library(survival)
  library(jsonlite)
})

set.seed(1)

FIXTURES_DIR <- "tests/fixtures"
dir.create(FIXTURES_DIR, showWarnings = FALSE, recursive = TRUE)

# Helper: shared synthetic Gaussian problem.
make_ls_problem <- function(n = 200, p = 20, seed = 1) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- numeric(p)
  beta[1:3] <- c(1.5, -2.0, 0.8)
  y <- X %*% beta + 0.1 * rnorm(n)
  list(X = X, y = as.numeric(y), beta_true = beta, n = n, p = p)
}

# Helper: synthetic logistic problem.
make_logistic_problem <- function(n = 300, p = 15, seed = 2) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- numeric(p)
  beta[1:3] <- c(1.5, -1.0, 0.8)
  eta <- X %*% beta
  prob <- 1 / (1 + exp(-eta))
  y <- as.numeric(runif(n) < prob)
  list(X = X, y = y, beta_true = beta, n = n, p = p)
}

# Helper: synthetic Cox PH problem with right censoring.
make_cox_problem <- function(n = 300, p = 15, seed = 3) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  beta <- numeric(p)
  beta[1:3] <- c(0.7, -0.5, 0.3)
  eta <- X %*% beta
  rate_t <- exp(eta)
  t_event <- -log(runif(n)) / rate_t
  t_cens <- rexp(n, rate = 0.5)
  time <- pmin(t_event, t_cens)
  event <- as.numeric(t_event <= t_cens)
  list(X = X, time = as.numeric(time), event = event,
       beta_true = beta, n = n, p = p)
}

# Helper: synthetic group-structured problem.
make_group_problem <- function(n = 200, n_groups = 6, group_size = 4, seed = 4) {
  set.seed(seed)
  p <- n_groups * group_size
  X <- matrix(rnorm(n * p), n, p)
  groups <- rep(seq_len(n_groups), each = group_size)
  beta <- numeric(p)
  active_groups <- c(1, 3)
  for (g in active_groups) {
    idx <- which(groups == g)
    beta[idx] <- rnorm(length(idx), 0, 0.6)
  }
  y <- X %*% beta + 0.1 * rnorm(n)
  list(X = X, y = as.numeric(y), groups = groups,
       active_groups = active_groups, beta_true = beta,
       n = n, p = p)
}

# Serialize a fit to JSON. R matrices become lists-of-rows in JSON,
# which jsonlite handles via `as.matrix()` + `toJSON(matrix=...)`.
write_fixture <- function(name, payload) {
  path <- file.path(FIXTURES_DIR, paste0(name, ".json"))
  json <- toJSON(payload, matrix = "rowmajor", auto_unbox = TRUE,
                 digits = 17, na = "string")
  writeLines(json, path)
  cat(sprintf("wrote %s\n", path))
}

# ---- Fixture 1: glmnet gaussian lasso path ----
{
  prob <- make_ls_problem(seed = 11)
  fit <- glmnet(prob$X, prob$y, family = "gaussian", alpha = 1,
                standardize = TRUE, intercept = TRUE,
                control = list(thresh = 1e-10), nlambda = 30)
  # glmnet stores coef as (p+1) × n_lambda sparseMatrix; first row
  # is intercept.
  coef_full <- as.matrix(coef(fit))
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])  # (n_lambda, p)
  write_fixture("glmnet_lasso_gaussian", list(
    package = "glmnet",
    package_version = as.character(packageVersion("glmnet")),
    family = "gaussian",
    alpha = 1.0,
    standardize = TRUE,
    intercept = TRUE,
    thresh = 1e-10,
    n = prob$n, p = prob$p,
    seed = 11,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 2: ncvreg gaussian MCP path ----
{
  prob <- make_ls_problem(seed = 13)
  fit <- ncvreg(prob$X, prob$y, family = "gaussian",
                penalty = "MCP", gamma = 3.0,
                eps = 1e-10, max.iter = 50000,
                nlambda = 30)
  # ncvreg's beta is (p+1) × n_lambda; first row is intercept.
  coef_full <- as.matrix(fit$beta)
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("ncvreg_mcp_gaussian", list(
    package = "ncvreg",
    package_version = as.character(packageVersion("ncvreg")),
    family = "gaussian",
    penalty = "MCP",
    gamma = 3.0,
    eps = 1e-10,
    n = prob$n, p = prob$p,
    seed = 13,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 3: ncvreg gaussian SCAD path ----
{
  prob <- make_ls_problem(seed = 17)
  fit <- ncvreg(prob$X, prob$y, family = "gaussian",
                penalty = "SCAD", gamma = 3.7,
                eps = 1e-10, max.iter = 50000,
                nlambda = 30)
  coef_full <- as.matrix(fit$beta)
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("ncvreg_scad_gaussian", list(
    package = "ncvreg",
    package_version = as.character(packageVersion("ncvreg")),
    family = "gaussian",
    penalty = "SCAD",
    a = 3.7,
    eps = 1e-10,
    n = prob$n, p = prob$p,
    seed = 17,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 4: grpreg gaussian group lasso path ----
{
  prob <- make_group_problem(seed = 19)
  fit <- grpreg(prob$X, prob$y, group = prob$groups,
                family = "gaussian", penalty = "grLasso",
                eps = 1e-10, max.iter = 50000,
                nlambda = 30)
  coef_full <- as.matrix(fit$beta)
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("grpreg_grlasso_gaussian", list(
    package = "grpreg",
    package_version = as.character(packageVersion("grpreg")),
    family = "gaussian",
    penalty = "grLasso",
    eps = 1e-10,
    n = prob$n, p = prob$p,
    n_groups = max(prob$groups),
    seed = 19,
    X = prob$X, y = prob$y,
    groups = prob$groups,
    beta_true = prob$beta_true,
    # grpreg defaults group.multiplier = sqrt(group_size).
    group_multiplier = sqrt(as.numeric(table(prob$groups))),
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 5: grpreg gaussian group MCP path ----
{
  prob <- make_group_problem(seed = 23)
  fit <- grpreg(prob$X, prob$y, group = prob$groups,
                family = "gaussian", penalty = "grMCP",
                gamma = 3.0,
                eps = 1e-10, max.iter = 50000,
                nlambda = 30)
  coef_full <- as.matrix(fit$beta)
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("grpreg_grmcp_gaussian", list(
    package = "grpreg",
    package_version = as.character(packageVersion("grpreg")),
    family = "gaussian",
    penalty = "grMCP",
    gamma = 3.0,
    eps = 1e-10,
    n = prob$n, p = prob$p,
    n_groups = max(prob$groups),
    seed = 23,
    X = prob$X, y = prob$y,
    groups = prob$groups,
    beta_true = prob$beta_true,
    group_multiplier = sqrt(as.numeric(table(prob$groups))),
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 6: glmnet binomial lasso path ----
{
  prob <- make_logistic_problem(seed = 29)
  fit <- glmnet(prob$X, prob$y, family = "binomial", alpha = 1,
                standardize = TRUE, intercept = TRUE,
                control = list(thresh = 1e-10), nlambda = 30)
  coef_full <- as.matrix(coef(fit))
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("glmnet_lasso_binomial", list(
    package = "glmnet",
    package_version = as.character(packageVersion("glmnet")),
    family = "binomial",
    alpha = 1.0,
    standardize = TRUE,
    intercept = TRUE,
    thresh = 1e-10,
    n = prob$n, p = prob$p,
    seed = 29,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 7: ncvreg binomial MCP path ----
{
  prob <- make_logistic_problem(seed = 31)
  fit <- ncvreg(prob$X, prob$y, family = "binomial",
                penalty = "MCP", gamma = 3.0,
                eps = 1e-10, max.iter = 50000,
                nlambda = 30)
  coef_full <- as.matrix(fit$beta)
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("ncvreg_mcp_binomial", list(
    package = "ncvreg",
    package_version = as.character(packageVersion("ncvreg")),
    family = "binomial",
    penalty = "MCP",
    gamma = 3.0,
    eps = 1e-10,
    n = prob$n, p = prob$p,
    seed = 31,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 8: glmnet cox lasso path ----
{
  prob <- make_cox_problem(seed = 37)
  surv_y <- Surv(prob$time, prob$event)
  fit <- glmnet(prob$X, surv_y, family = "cox", alpha = 1,
                standardize = TRUE, cox.ties = "breslow",
                control = list(thresh = 1e-10), nlambda = 30)
  # No intercept for Cox.
  coefs <- t(as.matrix(coef(fit)))
  write_fixture("glmnet_lasso_cox", list(
    package = "glmnet",
    package_version = as.character(packageVersion("glmnet")),
    family = "cox",
    alpha = 1.0,
    standardize = TRUE,
    thresh = 1e-10,
    n = prob$n, p = prob$p,
    seed = 37,
    X = prob$X, time = prob$time, event = prob$event,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs
  ))
}

# ============================================================
# M14c.3 — at-scale fixtures (n=500, p=100)
# ============================================================
#
# A mid-tier set covering the same canonical (datafit × penalty)
# combinations as the small-tier above, at a size that exercises
# the path solvers under more realistic dynamics (sparse active
# sets, multiple outer iters per λ). Looser tolerances per
# scenario (the test side asserts `smallest_lambda_atol=1e-3`
# instead of `1e-5` and an `active_set_fuzz_frac` of 0.15) because
# LLA local-min divergence on nonconvex problems widens as p grows.
#
# Size pick: n=500, p=100 keeps each committed JSON ~1 MB
# uncompressed (50,000 floats in X + n_lambdas × p coefs); larger
# tiers would force a separate artifact-server pipeline. The
# committed mid-tier is enough to catch any "tolerance/scale-
# dependent" regression that the small fixtures miss.

# Helper: at-scale Gaussian problem (n=500, p=100, ~8 active features).
make_ls_problem_mid <- function(seed = 401) {
  set.seed(seed)
  n <- 500
  p <- 100
  X <- matrix(rnorm(n * p), n, p)
  beta <- numeric(p)
  beta[1:8] <- c(2.0, -1.5, 1.2, -0.9, 0.8, -0.6, 0.5, -0.4)
  y <- X %*% beta + 0.2 * rnorm(n)
  list(X = X, y = as.numeric(y), beta_true = beta, n = n, p = p)
}

# Helper: at-scale logistic problem.
make_logistic_problem_mid <- function(seed = 411) {
  set.seed(seed)
  n <- 500
  p <- 100
  X <- matrix(rnorm(n * p), n, p)
  beta <- numeric(p)
  beta[1:6] <- c(1.5, -1.2, 1.0, -0.8, 0.6, -0.4)
  eta <- X %*% beta
  prob <- 1 / (1 + exp(-eta))
  y <- as.numeric(runif(n) < prob)
  list(X = X, y = y, beta_true = beta, n = n, p = p)
}

# ---- Fixture 9: glmnet gaussian lasso path, mid scale ----
{
  prob <- make_ls_problem_mid(seed = 401)
  fit <- glmnet(prob$X, prob$y, family = "gaussian", alpha = 1,
                standardize = TRUE, intercept = TRUE,
                control = list(thresh = 1e-10), nlambda = 30)
  coef_full <- as.matrix(coef(fit))
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("glmnet_lasso_gaussian_mid", list(
    package = "glmnet",
    package_version = as.character(packageVersion("glmnet")),
    family = "gaussian",
    alpha = 1.0,
    standardize = TRUE,
    intercept = TRUE,
    thresh = 1e-10,
    scale = "mid",
    n = prob$n, p = prob$p,
    seed = 401,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 10: ncvreg gaussian MCP path, mid scale ----
{
  prob <- make_ls_problem_mid(seed = 403)
  fit <- ncvreg(prob$X, prob$y, family = "gaussian",
                penalty = "MCP", gamma = 3.0,
                eps = 1e-10, max.iter = 50000,
                nlambda = 30)
  coef_full <- as.matrix(fit$beta)
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("ncvreg_mcp_gaussian_mid", list(
    package = "ncvreg",
    package_version = as.character(packageVersion("ncvreg")),
    family = "gaussian",
    penalty = "MCP",
    gamma = 3.0,
    eps = 1e-10,
    scale = "mid",
    n = prob$n, p = prob$p,
    seed = 403,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

# ---- Fixture 11: glmnet binomial lasso path, mid scale ----
{
  prob <- make_logistic_problem_mid(seed = 411)
  fit <- glmnet(prob$X, prob$y, family = "binomial", alpha = 1,
                standardize = TRUE, intercept = TRUE,
                control = list(thresh = 1e-10), nlambda = 30)
  coef_full <- as.matrix(coef(fit))
  intercepts <- coef_full[1, ]
  coefs <- t(coef_full[-1, ])
  write_fixture("glmnet_lasso_binomial_mid", list(
    package = "glmnet",
    package_version = as.character(packageVersion("glmnet")),
    family = "binomial",
    alpha = 1.0,
    standardize = TRUE,
    intercept = TRUE,
    thresh = 1e-10,
    scale = "mid",
    n = prob$n, p = prob$p,
    seed = 411,
    X = prob$X, y = prob$y,
    beta_true = prob$beta_true,
    lambdas = as.numeric(fit$lambda),
    coefs = coefs,
    intercepts = as.numeric(intercepts)
  ))
}

cat("\nAll fixtures generated.\n")
