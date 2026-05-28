#' Bind Two Arrays Along the First Dimension
#'
#' Concatenates two arrays along the first dimension (samples axis).
#'
#' @param a First array
#' @param b Second array (must match dimensions of \code{a} except along dim 1)
#' @return Combined array
#' @keywords internal
abind_first <- function(a, b) {
  da <- dim(a)
  db <- dim(b)
  if (length(da) != length(db) || !all(da[-1] == db[-1])) {
    stop("validation_data dimensions do not match training data dimensions")
  }
  nd <- length(da)
  if (nd == 1L) {
    return(c(a, b))
  }
  # Concatenate along dim 1. R arrays are column-major, so c(a, b) only works
  # when the trailing dims are 1. General case: move dim 1 to the last axis
  # (where concatenation is contiguous), c(), then move it back.
  perm_to   <- c(seq_len(nd)[-1L], 1L)   # 1 -> last
  perm_back <- c(nd, seq_len(nd - 1L))   # last -> 1
  at <- aperm(a, perm_to)
  bt <- aperm(b, perm_to)
  combined <- array(c(at, bt), dim = c(da[-1], da[1] + db[1]))
  aperm(combined, perm_back)
}

#' Slice an Array or Matrix Along Its First Dimension
#'
#' Selects rows \code{idx} along the first (sample) axis, keeping all other
#' dimensions intact, regardless of how many dimensions the array has.
#'
#' @param x A matrix or array.
#' @param idx Integer indices into the first dimension.
#' @return \code{x} restricted to \code{idx} along dim 1 (\code{drop = FALSE}).
#' @keywords internal
slice_first_dim <- function(x, idx) {
  nd <- length(dim(x))
  if (nd <= 1L) {
    return(x[idx])
  }
  # Build x[idx, , , ..., drop = FALSE] with the right number of empty args.
  args <- c(list(x, idx), rep(list(quote(expr = )), nd - 1L), list(drop = FALSE))
  do.call(`[`, args)
}

#' Create Context with Auto-sizing
#'
#' Creates a context with automatically calculated size based on planned tensors.
#'
#' @param ... Named arguments with tensor dimensions (integer vectors)
#' @param extra_mb Extra megabytes to add (default: 10)
#' @param type Tensor type (default: GGML_TYPE_F32)
#' @param no_alloc If TRUE, don't allocate memory for tensors (default: FALSE)
#' @return GGML context
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init_auto(mat1 = c(1000L, 1000L), mat2 = c(1000L, 1000L))
#' ggml_free(ctx)
#' }
ggml_init_auto <- function(..., extra_mb = 10, type = GGML_TYPE_F32, no_alloc = FALSE) {
  tensors <- list(...)

  total_bytes <- 0

  for (t in tensors) {
    if (length(t) == 1) {
      # 1D tensor
      total_bytes <- total_bytes + ggml_estimate_memory(type, t)
    } else if (length(t) == 2) {
      # 2D tensor
      total_bytes <- total_bytes + ggml_estimate_memory(type, t[1], t[2])
    } else if (length(t) == 3) {
      # 3D tensor
      total_bytes <- total_bytes + ggml_estimate_memory(type, t[1], t[2], t[3])
    } else if (length(t) == 4) {
      # 4D tensor
      total_bytes <- total_bytes + ggml_estimate_memory(type, t[1], t[2], t[3], t[4])
    }
  }

  # Add extra space
  total_bytes <- total_bytes + extra_mb * 1024 * 1024

  ggml_init(as.integer(total_bytes), no_alloc = no_alloc)
}

#' Execute with Temporary Context
#' 
#' Creates a temporary context, executes code, and frees it automatically.
#' Useful when you need to create large temporary tensors.
#'
#' @param mem_size Context memory size in bytes
#' @param expr Expression to evaluate with the temporary context
#' @return Result of the expression
#' @export
#' @examples
#' \donttest{
#' # Create tensors in temporary context
#' result <- ggml_with_temp_ctx(1024 * 1024, {
#'   a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#'   ggml_set_f32(a, 1:10)
#'   ggml_get_f32(a)
#' })
#' }
ggml_with_temp_ctx <- function(mem_size, expr) {
  ctx <- ggml_init(mem_size)
  on.exit(ggml_free(ctx), add = TRUE)
  
  # Evaluate expression in calling environment
  # but with 'ctx' available
  eval(substitute(expr), envir = list(ctx = ctx), enclos = parent.frame())
}
