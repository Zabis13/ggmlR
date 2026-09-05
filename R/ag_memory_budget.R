# Training memory budget for the ag_* path, from parameter shapes alone.
#
# ggml_estimate_memory() (R/memory.R) sizes ONE tensor in a ggml buffer, where
# dtype is real and quantised types have block sizes. This is a different
# question with a different answer: what does TRAINING a model cost, counting
# the four things that are alive at once --
#
#   weights + gradients + optimizer states + activations
#
# and on the ag_* path those live in R, not in a ggml buffer.
#
# The single most useful thing this function says is that the budget is twice
# what an f32 estimate suggests. Everything on the backward side of path B is R
# double: backward() accumulates $grad in double closures, and optimizer m/v are
# R matrices (established when "optimizer states в f32 при f16" and "loss
# scaling" were closed -- both turned out to be already-true by construction,
# for this reason). ag_dtype() applies only to forward tensors being uploaded
# into a ggml buffer; it never reaches gradients or optimizer state. So a
# parameter of n elements costs 8n bytes per copy, not 4n.
#
# Shapes rather than a model object. Taking a compiled model or a list of
# ag_param would mean introspecting a format that differs between the two paths
# -- path A counts activations inside the ggml graph, path B on the tape -- and
# there is no caller for that yet. Shapes work before a model exists, which is
# when the question "will this fit" is actually asked.

# Bytes per scalar on the ag_* path. Not a parameter: making it one would imply
# the path can store gradients in anything else, which is exactly the confusion
# this file exists to prevent.
.AG_BYTES_PER_SCALAR <- 8

# Optimizer state multiplier: how many extra full-size copies of every
# parameter the optimizer keeps.
#
#   adam     m and v                          -> 2
#   sgd      velocity, only with momentum > 0  -> 1
#   sgd_plain no state                         -> 0
#
# This is the single source of truth for that number; nothing else in the
# package should restate it.
.ag_opt_state_multiplier <- function(optimizer) {
  switch(optimizer,
         adam      = 2,
         sgd       = 1,
         sgd_plain = 0,
         stop("ggmlR: unknown optimizer '", optimizer,
              "'. Use \"adam\", \"sgd\" or \"sgd_plain\".", call. = FALSE))
}

#' Estimate the memory a training run needs
#'
#' Adds up the four things that are alive at the same time during training on
#' the autograd (\code{ag_*}) path: the weights, their gradients, the
#' optimizer's per-parameter state, and the activations held by the tape until
#' \code{zero_grad()} clears it.
#'
#' Everything is counted at 8 bytes per scalar, because that is what the path
#' actually uses: gradients are accumulated by R closures in double, and
#' optimizer moments are R matrices. \code{\link{ag_dtype}} affects only
#' forward tensors uploaded into a ggml buffer and does not reach either. An
#' estimate made in f32 would be half the real figure, so the returned list
#' carries a \code{note} saying so.
#'
#' @param shapes List of parameter shapes, each a numeric vector of dimensions
#'   (e.g. \code{list(c(256, 128), c(128))}), or a single such vector.
#' @param batch_size Batch size the activations are sized for. Activations
#'   scale with it; weights and optimizer state do not.
#' @param optimizer One of \code{"adam"} (keeps m and v), \code{"sgd"} (keeps a
#'   velocity, i.e. momentum > 0) or \code{"sgd_plain"} (no state).
#' @param activation_frac Activations as a fraction of the weight bytes, per
#'   unit of batch. The default 0.01 reproduces the measured tape composition
#'   for a dense stack (\code{inst/scripts/measure_ag_tape_memory.R}: activations
#'   were 19\% of a 4-layer 128-wide tape at batch 32). Pass your own if you
#'   have measured your model with \code{\link{ag_tape_memory}}, which is always
#'   more accurate than this scaling.
#' @param quiet Suppress the printed report and return the figures only. Note
#'   that the double-vs-f32 caveat is part of that report, so a caller passing
#'   \code{quiet = TRUE} takes on reading it from the \code{note} field.
#' @return Invisibly, a list of byte counts: \code{params}, \code{weights},
#'   \code{gradients}, \code{optimizer}, \code{activations}, \code{total}, plus
#'   \code{bytes_per_scalar} and \code{note}. Printed as a report.
#' @seealso \code{\link{ag_tape_memory}} measures a real tape instead of
#'   estimating one; \code{\link{ggml_estimate_memory}} sizes a single tensor in
#'   a ggml buffer, where dtype does apply.
#' @export
#' @examples
#' \donttest{
#' # a 3-layer MLP, 512 wide, trained with Adam at batch 64
#' ag_estimate_training_memory(
#'   shapes = list(c(512, 512), c(512, 512), c(512, 512)),
#'   batch_size = 64)
#' }
ag_estimate_training_memory <- function(shapes,
                                        batch_size = 32L,
                                        optimizer = c("adam", "sgd", "sgd_plain"),
                                        activation_frac = 0.01,
                                        quiet = FALSE) {
  optimizer <- match.arg(optimizer)
  if (is.numeric(shapes)) shapes <- list(shapes)
  if (!is.list(shapes) || length(shapes) == 0L)
    stop("ggmlR: `shapes` must be a non-empty list of dimension vectors.",
         call. = FALSE)

  n_par <- 0
  for (s in shapes) {
    if (!is.numeric(s) || length(s) == 0L || any(!is.finite(s)) || any(s <= 0))
      stop("ggmlR: each shape must be a vector of positive dimensions.",
           call. = FALSE)
    n_par <- n_par + prod(as.double(s))
  }

  b <- .AG_BYTES_PER_SCALAR
  k <- .ag_opt_state_multiplier(optimizer)

  w_bytes <- n_par * b
  g_bytes <- n_par * b            # one gradient per parameter, same size
  o_bytes <- n_par * b * k
  # Activations are the one term that is not a fixed multiple of the parameter
  # count: they scale with the batch, which is why the measured tape share
  # ranged from 1.4% at batch 1 to 31.6% at batch 128 on the same model.
  a_bytes <- w_bytes * activation_frac * as.double(batch_size)

  total <- w_bytes + g_bytes + o_bytes + a_bytes

  note <- paste0(
    "path B stores gradients and optimizer state as R double (8 bytes); ",
    "an f32 estimate would be half this. ag_dtype() does not apply.")

  mb <- function(x) x / 1024^2
  if (!isTRUE(quiet)) {
    cat(sprintf("Training memory estimate (%s, batch %d)\n",
                optimizer, as.integer(batch_size)))
    cat(sprintf("  parameters      : %s\n", format(n_par, big.mark = " ")))
    cat(sprintf("  weights         : %8.2f MB\n", mb(w_bytes)))
    cat(sprintf("  gradients       : %8.2f MB\n", mb(g_bytes)))
    cat(sprintf("  optimizer (x%d)  : %8.2f MB\n", k, mb(o_bytes)))
    cat(sprintf("  activations     : %8.2f MB\n", mb(a_bytes)))
    cat(sprintf("  total           : %8.2f MB\n", mb(total)))
    cat("  note: ", note, "\n", sep = "")
  }

  invisible(list(params           = n_par,
                 weights          = w_bytes,
                 gradients        = g_bytes,
                 optimizer        = o_bytes,
                 activations      = a_bytes,
                 total            = total,
                 bytes_per_scalar = b,
                 note             = note))
}
