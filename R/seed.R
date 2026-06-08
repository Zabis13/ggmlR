# Seed / determinism API ------------------------------------------------------
#
# All stochastic behaviour in ggmlR (weight initialisation, dropout masks and
# data shuffling) is generated in R via the base R RNG (runif / rnorm /
# sample). There is no random number generation on the C / Vulkan side. Fixing
# the R seed therefore makes the *structure* of a run reproducible: identical
# starting weights, identical dropout masks and identical batch ordering.

#' Set the random seed for reproducible ggmlR runs
#'
#' Fixes the random seed used by ggmlR for everything that is stochastic:
#' \itemize{
#'   \item weight initialisation (sequential, functional and autograd layers),
#'   \item dropout masks (training-time),
#'   \item data shuffling in the autograd dataloader / training loops.
#' }
#'
#' This is a thin wrapper around \code{\link[base]{set.seed}}: all randomness in
#' ggmlR is produced by the base R RNG, so a fixed seed gives identical starting
#' weights, dropout masks and batch ordering across runs. It is the single point
#' of control used by the \pkg{mlr3} learners (\code{seed} hyperparameter) and
#' the \pkg{parsnip} \code{"ggml"} engine (\code{seed} engine argument).
#'
#' \strong{GPU note}: this controls the random *inputs* to the computation, not
#' the floating-point arithmetic itself. GPU (Vulkan) kernels are run-to-run
#' stable on a given device/driver for the standard forward/backward paths, but
#' ggmlR does not guarantee \emph{bit-for-bit} identical results across
#' different devices, drivers or backends (CPU vs Vulkan). Reproducibility is at
#' the level of training dynamics, not exact bits.
#'
#' @param seed A single integer (or value coercible to integer) used as the RNG
#'   seed. \code{NULL} is a no-op, which lets callers thread an optional seed
#'   argument through without special-casing it.
#'
#' @return Invisibly returns \code{seed}.
#'
#' @examples
#' ggml_set_seed(42)
#' a <- runif(3)
#' ggml_set_seed(42)
#' b <- runif(3)
#' identical(a, b)  # TRUE
#'
#' @seealso \code{\link[base]{set.seed}}
#' @export
ggml_set_seed <- function(seed) {
  if (is.null(seed)) return(invisible(NULL))
  if (length(seed) != 1L || is.na(seed)) {
    stop("`seed` must be a single non-NA integer (or NULL).", call. = FALSE)
  }
  set.seed(as.integer(seed))
  invisible(as.integer(seed))
}
