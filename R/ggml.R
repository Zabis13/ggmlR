#' @useDynLib ggmlR, .registration = TRUE
#' @keywords internal
"_PACKAGE"

#' Initialize GGML context
#' @param mem_size Memory size in bytes
#' @param no_alloc If TRUE, don't allocate memory for tensors (default: FALSE)
#' @return GGML context pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_free(ctx)
#' }
ggml_init <- function(mem_size = 16 * 1024 * 1024, no_alloc = FALSE) {
  .Call("R_ggml_init", as.numeric(mem_size), as.logical(no_alloc), PACKAGE = "ggmlR")
}

#' Free GGML context
#' @param ctx Context pointer
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_free(ctx)
#' }
ggml_free <- function(ctx) {
  invisible(.Call("R_ggml_free", ctx, PACKAGE = "ggmlR"))
}

#' Create 1D tensor
#' @param ctx GGML context
#' @param type Data type (e.g., GGML_TYPE_F32)
#' @param ne0 Number of elements
#' @return Tensor pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_nelements(tensor)
#' ggml_free(ctx)
#' }
ggml_new_tensor_1d <- function(ctx, type, ne0) {
  .Call("R_ggml_new_tensor_1d", ctx, as.integer(type), as.numeric(ne0), PACKAGE = "ggmlR")
}

#' Create 2D tensor
#' @param ctx GGML context
#' @param type Data type (e.g., GGML_TYPE_F32)
#' @param ne0 Number of rows
#' @param ne1 Number of columns
#' @return Tensor pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' tensor <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' ggml_nelements(tensor)
#' ggml_free(ctx)
#' }
ggml_new_tensor_2d <- function(ctx, type, ne0, ne1) {
  .Call("R_ggml_new_tensor_2d", ctx, as.integer(type), as.numeric(ne0), as.numeric(ne1), PACKAGE = "ggmlR")
}

#' Set F32 data
#' @param tensor Tensor pointer
#' @param data Numeric vector of values to set
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(tensor, c(1, 2, 3, 4, 5))
#' ggml_get_f32(tensor)
#' ggml_free(ctx)
#' }
ggml_set_f32 <- function(tensor, data) {
  invisible(.Call("R_ggml_set_f32", tensor, as.numeric(data), PACKAGE = "ggmlR"))
}

#' Get F32 data
#' @param tensor Tensor pointer
#' @return Numeric vector with tensor values
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(tensor, c(1, 2, 3, 4, 5))
#' ggml_get_f32(tensor)
#' ggml_free(ctx)
#' }
ggml_get_f32 <- function(tensor) {
  .Call("R_ggml_get_f32", tensor, PACKAGE = "ggmlR")
}

#' Add tensors
#' @param ctx GGML context
#' @param a First tensor
#' @param b Second tensor (same shape as a)
#' @return Tensor representing the addition operation
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(5, 4, 3, 2, 1))
#' result <- ggml_add(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_add <- function(ctx, a, b) {
  .Call("R_ggml_add", ctx, a, b, PACKAGE = "ggmlR")
}

#' Multiply tensors
#' @param ctx GGML context
#' @param a First tensor
#' @param b Second tensor (same shape as a)
#' @return Tensor representing the multiplication operation
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(2, 2, 2, 2, 2))
#' result <- ggml_mul(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_mul <- function(ctx, a, b) {
  .Call("R_ggml_mul", ctx, a, b, PACKAGE = "ggmlR")
}

#' Build forward expand
#'
#' Builds a computation graph from the output tensor, expanding backwards
#' to include all dependencies.
#'
#' @param ctx GGML context
#' @param tensor Output tensor to build graph for
#' @return Graph pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(5, 4, 3, 2, 1))
#' result <- ggml_add(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_build_forward_expand <- function(ctx, tensor) {
  .Call("R_ggml_build_forward_expand", ctx, tensor, PACKAGE = "ggmlR")
}

#' Compute graph
#'
#' Executes all operations in the computation graph.
#'
#' @param ctx GGML context
#' @param graph Graph pointer from ggml_build_forward_expand
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' result <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_graph_compute <- function(ctx, graph) {
  invisible(.Call("R_ggml_graph_compute", ctx, graph, PACKAGE = "ggmlR"))
}

#' Get GGML version
#' @return Character string with GGML version
#' @export
#' @examples
#' \dontrun{
#' ggml_version()
#' }
ggml_version <- function() {
  .Call("R_ggml_version", PACKAGE = "ggmlR")
}

#' Test GGML
#'
#' Runs GGML library self-test and prints version info.
#'
#' @return TRUE if test passed
#' @export
#' @examples
#' \dontrun{
#' ggml_test()
#' }
ggml_test <- function() {
  .Call("R_ggml_test", PACKAGE = "ggmlR")
}

#' Get number of elements
#' @param tensor Tensor pointer
#' @return Integer number of elements
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' tensor <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' ggml_nelements(tensor)
#' ggml_free(ctx)
#' }
ggml_nelements <- function(tensor) {
  .Call("R_ggml_nelements", tensor, PACKAGE = "ggmlR")
}

#' Get number of bytes
#' @param tensor Tensor pointer
#' @return Integer number of bytes
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' tensor <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_nbytes(tensor)
#' ggml_free(ctx)
#' }
ggml_nbytes <- function(tensor) {
  .Call("R_ggml_nbytes", tensor, PACKAGE = "ggmlR")
}

#' Check if GGML is available
#' @return TRUE if GGML library is loaded
#' @export
#' @examples
#' \dontrun{
#' ggml_is_available()
#' }
ggml_is_available <- function() {
  TRUE
}

# GGML type constants are defined in R/tensors.R

#' Reset GGML Context
#'
#' Clears all tensor allocations in the context memory pool.
#' The context can be reused without recreating it.
#' This is more efficient than free + init for temporary operations.
#'
#' @param ctx GGML context pointer
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' ggml_reset(ctx)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 200)
#' ggml_free(ctx)
#' }
ggml_reset <- function(ctx) {
  invisible(.Call("R_ggml_reset", ctx))
}

#' Initialize GGML Timer
#'
#' Initializes the GGML timing system. Call this once at the beginning
#' of the program before using ggml_time_ms() or ggml_time_us().
#'
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ggml_time_init()
#' start <- ggml_time_ms()
#' Sys.sleep(0.01)
#' elapsed <- ggml_time_ms() - start
#' }
ggml_time_init <- function() {
  invisible(.Call("R_ggml_time_init"))
}

#' Get Time in Milliseconds
#'
#' Returns the current time in milliseconds since the timer was initialized.
#'
#' @return Numeric value representing milliseconds
#' @export
#' @examples
#' \dontrun{
#' ggml_time_init()
#' start <- ggml_time_ms()
#' Sys.sleep(0.01)
#' elapsed <- ggml_time_ms() - start
#' }
ggml_time_ms <- function() {
  .Call("R_ggml_time_ms")
}

#' Get Time in Microseconds
#'
#' Returns the current time in microseconds since the timer was initialized.
#' More precise than ggml_time_ms() for micro-benchmarking.
#'
#' @return Numeric value representing microseconds
#' @export
#' @examples
#' \dontrun{
#' ggml_time_init()
#' start <- ggml_time_us()
#' Sys.sleep(0.001)
#' elapsed <- ggml_time_us() - start
#' }
ggml_time_us <- function() {
  .Call("R_ggml_time_us")
}

#' Get CPU Cycles
#'
#' Returns the current CPU cycle count. Useful for low-level benchmarking.
#'
#' @return Numeric value representing CPU cycles
#' @export
#' @examples
#' \dontrun{
#' ggml_cycles()
#' }
ggml_cycles <- function() {
  .Call("R_ggml_cycles")
}

#' Get CPU Cycles per Millisecond
#'
#' Returns an estimate of CPU cycles per millisecond.
#' Useful for converting cycle counts to time.
#'
#' @return Numeric value representing cycles per millisecond
#' @export
#' @examples
#' \dontrun{
#' ggml_cycles_per_ms()
#' }
ggml_cycles_per_ms <- function() {
  .Call("R_ggml_cycles_per_ms")
}
