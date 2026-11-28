# User-defined differentiable operations: ag_function(forward, backward).
#
# The analogue of torch's autograd_function. It is a thin public layer over
# ag_record(), the same mechanism every built-in closure op (ag_pow, the
# batch-norm module, ...) uses: forward runs in R on plain matrices, and the
# backward closure is recorded on the tape with the tensor inputs.
#
# Two rules the wrapper enforces so user code never sees internals:
#   * forward and backward get plain R matrices. A resident tensor is read with
#     .ag_data(); grad_output may arrive as a device handle on a resident tape,
#     and handles deliberately have no arithmetic, so it is converted with
#     .ag_as_matrix() before backward sees it.
#   * every gradient backward returns must have its input's shape (a plain
#     vector of the right length is reshaped); NULL means "no gradient".
#
# Cost: a node without a graph-backward rule sends the whole tape down the
# closure path (R/ag_backward_graph.R), and on the GPU forward downloads its
# inputs and uploads its result. Correct, but not the fast path.

.ag_function_ctx <- function() {
  ctx <- new.env(parent = emptyenv())
  ctx$saved <- list()
  ctx$save_for_backward <- function(...) {
    ctx$saved <- c(ctx$saved, list(...))
    invisible(NULL)
  }
  ctx
}

# Name every argument of a call to `forward` (minus its leading ctx) by the
# formal it binds to, the way R itself would match it.
.ag_function_match_args <- function(forward, args) {
  if (length(args) == 0L) return(args)
  call <- as.call(c(list(as.name("forward"), ctx = NULL),
                    lapply(seq_along(args), function(i) as.name(sprintf(".a%d", i)))))
  if (!is.null(names(args))) names(call)[-(1:2)] <- names(args)
  mc <- match.call(forward, call, expand.dots = TRUE)
  out <- vector("list", length(args))
  nms <- character(length(args))
  for (j in seq_along(mc)[-1L]) {
    if (!is.name(mc[[j]])) next               # the ctx placeholder (NULL)
    sym <- as.character(mc[[j]])
    if (!startsWith(sym, ".a")) next
    i <- as.integer(substring(sym, 3L))
    out[i] <- list(args[[i]])
    nms[i] <- names(mc)[j]
  }
  names(out) <- nms
  out
}

#' Define a differentiable operation from a forward and a backward function
#'
#' The analogue of torch's \code{autograd_function()}: as long as you can write
#' the gradient, any computation becomes an operation of the autograd tape.
#' \code{ag_function()} returns a function; calling it on \code{ag_tensor}s
#' runs \code{forward} and records \code{backward} for \code{\link{backward}}.
#'
#' \code{forward(ctx, ...)} receives the call's arguments, each
#' \code{ag_tensor} replaced by its value as a plain R matrix, and returns the
#' result as a numeric matrix (or vector, taken as a column). Anything
#' \code{backward} needs is stored on \code{ctx}: \code{ctx$save_for_backward(...)}
#' appends to the list \code{ctx$saved}, and any other field can be set directly
#' (\code{ctx$p <- p}). \code{ctx$needs_input_grad} is a named logical: which
#' tensor arguments require a gradient.
#'
#' \code{backward(ctx, grad_output)} receives the same \code{ctx} and the
#' gradient of the loss with respect to the output, a matrix of the output's
#' shape. It returns a list of gradients named after \code{forward}'s arguments
#' (each with that argument's shape; \code{NULL} for none). Arguments that are
#' not tensors, or that do not require a gradient, may be left out.
#'
#' The operation runs in R. On the GPU its inputs are downloaded and its result
#' uploaded, and a tape that contains it computes its backward through R
#' closures rather than as one device graph. It cannot be used inside
#' \code{\link{ag_capture}}.
#'
#' @param forward \code{function(ctx, ...)} computing the result.
#' @param backward \code{function(ctx, grad_output)} returning a named list of
#'   input gradients.
#' @return A function with the same arguments as \code{forward} minus
#'   \code{ctx}, returning an \code{ag_tensor}.
#' @seealso \code{\link{ag_gradcheck}} to check a hand-written backward.
#' @export
#' @examples
#' # x^p with a hand-written gradient
#' pow <- ag_function(
#'   forward = function(ctx, x, p) {
#'     ctx$save_for_backward(x = x)
#'     ctx$p <- p
#'     x^p
#'   },
#'   backward = function(ctx, grad_output) {
#'     x <- ctx$saved$x
#'     list(x = grad_output * ctx$p * x^(ctx$p - 1))
#'   })
#' x <- ag_param(matrix(c(1, 2, 3, 4), 2), device = "cpu")
#' with_grad_tape(loss <- ag_sum(pow(x, 3)))
#' backward(loss)
#' x$grad   # 3 * x^2
ag_function <- function(forward, backward) {
  if (!is.function(forward)) stop("forward must be a function(ctx, ...)")
  if (!is.function(backward)) stop("backward must be a function(ctx, grad_output)")
  if (length(formals(forward)) < 1L) stop("forward must take ctx as its first argument")
  if (length(formals(backward)) < 2L) stop("backward must take (ctx, grad_output)")
  force(forward); force(backward)

  function(...) {
    if (isTRUE(.ag_capture_reg$recording)) {
      stop("ggmlR: an ag_function() operation cannot be recorded by ag_capture: ",
           "its backward is an R closure, not a graph rule. Call it outside ag_capture.",
           call. = FALSE)
    }
    args <- .ag_function_match_args(forward, list(...))
    is_t <- vapply(args, is_ag_tensor, logical(1))
    tens <- args[is_t]
    if (any(names(tens) == "")) {
      stop("ag_function: every ag_tensor argument must bind to a named argument of forward")
    }
    device <- if (length(tens)) tens[[1L]]$device else .ag_device_state$device

    vals <- args
    for (nm in names(tens)) vals[[nm]] <- .ag_data(tens[[nm]])
    shapes <- lapply(vals[names(tens)], function(v) if (is.null(dim(v))) c(length(v), 1L) else dim(v))

    ctx <- .ag_function_ctx()
    ctx$needs_input_grad <- vapply(tens, function(t) isTRUE(t$requires_grad), logical(1))
    res <- do.call(forward, c(list(ctx), unname(vals[!nzchar(names(vals))]),
                              vals[nzchar(names(vals))]))
    if (!is.numeric(res) || (!is.null(dim(res)) && length(dim(res)) != 2L)) {
      stop("ag_function: forward must return a numeric matrix or vector")
    }
    if (is.null(dim(res))) res <- matrix(res, ncol = 1L)
    out <- ag_tensor(res, device = device)
    out$requires_grad <- any(ctx$needs_input_grad)

    if (out$requires_grad) {
      grad_fn <- function(grad_out) {
        g <- backward(ctx, .ag_as_matrix(grad_out))
        if (!is.list(g)) stop("ag_function: backward must return a list of gradients")
        unknown <- setdiff(names(g), names(tens))
        if (is.null(names(g)) || any(names(g) == "") || length(unknown)) {
          stop("ag_function: backward must return gradients named after forward's ",
               "tensor arguments (", paste(names(tens), collapse = ", "), ")",
               if (length(unknown)) paste0("; unknown: ", paste(unknown, collapse = ", ")))
        }
        for (nm in names(g)) {
          gi <- g[[nm]]
          if (is.null(gi)) next
          want <- shapes[[nm]]
          if (is.null(dim(gi)) && length(gi) == prod(want)) {
            gi <- matrix(as.numeric(gi), want[1L], want[2L])
          } else if (!identical(as.integer(dim(gi)), as.integer(want))) {
            stop(sprintf("ag_function: gradient for '%s' has shape (%s); expected (%s)", nm,
                         paste(if (is.null(dim(gi))) length(gi) else dim(gi), collapse = ", "),
                         paste(want, collapse = ", ")))
          }
          g[nm] <- list(gi)
        }
        g
      }
      out$grad_fn <- grad_fn
      ag_record(out, grad_fn, tens)
    }
    out
  }
}
