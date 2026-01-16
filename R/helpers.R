#' Create Context with Auto-sizing
#' 
#' Creates a context with automatically calculated size based on planned tensors
#'
#' @param ... Named arguments with tensor dimensions
#' @param extra_mb Extra megabytes to add (default: 10)
#' @param type Tensor type (default: GGML_TYPE_F32)
#' @return GGML context
#' @export
#' @examples
#' \dontrun{
#' # For two 1000x1000 matrices
#' ctx <- ggml_init_auto(mat1 = c(1000, 1000), mat2 = c(1000, 1000))
#' }
ggml_init_auto <- function(..., extra_mb = 10, type = GGML_TYPE_F32) {
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
  
  ggml_init(as.integer(total_bytes))
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
#' \dontrun{
#' # Create large matrix in temporary context
#' result <- ggml_with_temp_ctx(100 * 1024 * 1024, {
#'   a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3000, 3000)
#'   b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3000, 3000)
#'   ggml_set_f32(a, rnorm(9000000))
#'   ggml_set_f32(b, rnorm(9000000))
#'   ggml_cpu_add(a, b)
#' })
#' }
ggml_with_temp_ctx <- function(mem_size, expr) {
  ctx <- ggml_init(mem_size)
  on.exit(ggml_free(ctx), add = TRUE)
  
  # Evaluate expression in calling environment
  # but with 'ctx' available
  eval(substitute(expr), envir = list(ctx = ctx), enclos = parent.frame())
}
