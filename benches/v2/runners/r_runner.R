# r_runner.R — v2 R-side runner. Reads a feather workdir produced by
# benches.v2.runners._r_io.write_request, fits with one of {glmnet,
# ncvreg, grpreg, glasso}, and writes result.feather + result_meta.feather.
#
# Usage: Rscript r_runner.R <workdir>
#
# Files read:
#   <workdir>/config.feather       — scalars (package, penalty, family, tol, gamma, n, p, has_groups, ...)
#   <workdir>/X.feather            — n × p, columns x0..x{p-1}
#   <workdir>/y.feather            — n × 1
#   <workdir>/lambda.feather       — n_lambdas × 1
#   <workdir>/groups.feather       — optional, p × 1 (int64)
#
# Files written:
#   <workdir>/result.feather       — coef_path n_lambdas × p (columns x0..x{p-1})
#   <workdir>/result_meta.feather  — one-row table: fit_time_s, version, active_set_size, n_iter

suppressPackageStartupMessages({
  library(arrow)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("usage: Rscript r_runner.R <workdir>")
}
workdir <- args[[1L]]

# --- read request -----------------------------------------------------
config <- as.data.frame(read_feather(file.path(workdir, "config.feather")))
package <- as.character(config$package)
penalty <- as.character(config$penalty)
family  <- as.character(config$family)
tol     <- as.numeric(config$tol)
gamma   <- if ("gamma" %in% names(config) && !is.na(config$gamma)) as.numeric(config$gamma) else NULL
has_groups <- as.logical(config$has_groups)
p_expected <- as.integer(config$p)

# Read X efficiently: arrow::read_feather returns a tibble with x0..x{p-1}.
x_df <- read_feather(file.path(workdir, "X.feather"))
# Preserve column order (x0..x{p-1}); arrow keeps insertion order so this
# is a no-op in practice, but make it explicit.
x_cols <- sprintf("x%d", seq.int(0L, p_expected - 1L))
X <- as.matrix(x_df[, x_cols, drop = FALSE])
storage.mode(X) <- "double"
y <- as.numeric(read_feather(file.path(workdir, "y.feather"))$y)
lambdas <- as.numeric(read_feather(file.path(workdir, "lambda.feather"))$lambda)

groups <- NULL
if (isTRUE(has_groups)) {
  groups <- as.integer(read_feather(file.path(workdir, "groups.feather"))$group)
}

# Cox event status (0/1) is shipped as a sibling payload when family == "cox".
# Built into Surv(time, status) inside fit_glmnet.
has_status <- isTRUE(as.logical(config$has_status))
status <- NULL
if (has_status) {
  status <- as.integer(read_feather(file.path(workdir, "status.feather"))$status)
}

# --- dispatch ---------------------------------------------------------

map_family <- function(family, pkg) {
  switch(
    family,
    gaussian = "gaussian",
    logistic = if (pkg == "glmnet") "binomial" else "binomial",
    poisson  = "poisson",
    cox      = "cox",
    stop(sprintf("%s: unsupported family %s", pkg, family))
  )
}

fit_glmnet <- function() {
  library(glmnet)
  library(survival)
  glm_family <- map_family(family, "glmnet")
  alpha <- switch(penalty,
                  lasso = 1.0,
                  elastic_net = 0.5,
                  ridge = 0.0,
                  stop(sprintf("glmnet: unsupported penalty %s", penalty)))
  # Cox: build the Surv(time, status) response from y + the sibling
  # status payload (written by _r_io.write_request when family=="cox").
  if (family == "cox") {
    if (is.null(status)) {
      stop("Cox via glmnet requires a status payload; status.feather not written")
    }
    y_arg <- Surv(time = y, event = status)
  } else {
    y_arg <- y
  }
  t0 <- proc.time()
  fit <- glmnet(X, y_arg, family = glm_family, alpha = alpha, lambda = lambdas,
                thresh = tol, standardize = FALSE)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  # fit$beta is a sparse p × n_lambdas dgCMatrix (no intercept row for Cox).
  coef_mat <- as.matrix(fit$beta)
  list(coef_path = t(coef_mat),
       fit_time_s = elapsed,
       version = as.character(packageVersion("glmnet")),
       active_set_size = sum(coef_mat[, ncol(coef_mat)] != 0),
       n_iter = NA_integer_)
}

fit_ncvreg <- function() {
  library(ncvreg)
  ncv_penalty <- switch(penalty,
                        mcp = "MCP", scad = "SCAD", lasso = "lasso",
                        stop(sprintf("ncvreg: unsupported penalty %s", penalty)))
  ncv_family <- map_family(family, "ncvreg")
  args <- list(X = X, y = y, family = ncv_family, penalty = ncv_penalty,
               lambda = lambdas, eps = tol, returnX = FALSE)
  if (!is.null(gamma)) args$gamma <- gamma
  t0 <- proc.time()
  fit <- do.call(ncvreg, args)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  # fit$beta is (p+1) × n_lambdas; row 1 is the intercept.
  coef_mat <- as.matrix(fit$beta)[-1L, , drop = FALSE]
  list(coef_path = t(coef_mat),
       fit_time_s = elapsed,
       version = as.character(packageVersion("ncvreg")),
       active_set_size = sum(coef_mat[, ncol(coef_mat)] != 0),
       n_iter = if (!is.null(fit$iter)) sum(fit$iter) else NA_integer_)
}

fit_grpreg <- function() {
  library(grpreg)
  if (is.null(groups)) stop("grpreg: groups required")
  grp_penalty <- switch(penalty,
                        group_lasso = "grLasso",
                        group_mcp   = "grMCP",
                        group_scad  = "grSCAD",
                        stop(sprintf("grpreg: unsupported penalty %s", penalty)))
  grp_family <- map_family(family, "grpreg")
  t0 <- proc.time()
  fit <- grpreg(X, y, group = groups, family = grp_family,
                penalty = grp_penalty, lambda = lambdas, eps = tol)
  elapsed <- (proc.time() - t0)[["elapsed"]]
  coef_mat <- as.matrix(fit$beta)[-1L, , drop = FALSE]
  list(coef_path = t(coef_mat),
       fit_time_s = elapsed,
       version = as.character(packageVersion("grpreg")),
       active_set_size = sum(coef_mat[, ncol(coef_mat)] != 0),
       n_iter = if (!is.null(fit$iter)) sum(fit$iter) else NA_integer_)
}

fit_glasso <- function() {
  library(glasso)
  if (family != "gaussian_inv_cov") stop("glasso: only gaussian_inv_cov supported")
  # X is the sample covariance matrix here; the Python side computes it.
  S <- X
  t0 <- proc.time()
  coefs <- vector("list", length(lambdas))
  for (k in seq_along(lambdas)) {
    fit <- glasso(S, rho = lambdas[k], thr = tol, maxit = 1000L)
    coefs[[k]] <- as.numeric(fit$wi)   # flatten p×p inverse covariance
  }
  elapsed <- (proc.time() - t0)[["elapsed"]]
  coef_mat <- do.call(cbind, coefs)   # (p*p) × n_lambdas
  list(coef_path = t(coef_mat),
       fit_time_s = elapsed,
       version = as.character(packageVersion("glasso")),
       active_set_size = sum(coef_mat[, ncol(coef_mat)] != 0),
       n_iter = NA_integer_)
}

result <- switch(
  package,
  glmnet = fit_glmnet(),
  ncvreg = fit_ncvreg(),
  grpreg = fit_grpreg(),
  glasso = fit_glasso(),
  stop(sprintf("unknown R package: %s", package))
)

# --- write response ---------------------------------------------------
# Coefficient path as n_lambdas × p table with columns x0..x{p-1}.
coef_mat <- result$coef_path
stopifnot(is.matrix(coef_mat))
p_out <- ncol(coef_mat)
colnames(coef_mat) <- sprintf("x%d", seq.int(0L, p_out - 1L))
write_feather(as.data.frame(coef_mat),
              file.path(workdir, "result.feather"),
              compression = "uncompressed")

meta <- data.frame(
  fit_time_s = result$fit_time_s,
  version    = result$version,
  active_set_size = as.integer(result$active_set_size),
  n_iter     = as.integer(if (is.na(result$n_iter)) NA_integer_ else result$n_iter),
  stringsAsFactors = FALSE
)
write_feather(meta, file.path(workdir, "result_meta.feather"),
              compression = "uncompressed")
