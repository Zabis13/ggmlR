#' @useDynLib ggmlR, .registration = TRUE
#' @keywords internal
"_PACKAGE"

#' Initialize GGML context
#' @param mem_size Memory size in bytes
#' @export
ggml_init <- function(mem_size = 16 * 1024 * 1024) {
  .Call("R_ggml_init", as.integer(mem_size), PACKAGE = "ggmlR")
}

#' Free GGML context
#' @param ctx Context pointer
#' @export
ggml_free <- function(ctx) {
  invisible(.Call("R_ggml_free", ctx, PACKAGE = "ggmlR"))
}

#' Create 1D tensor
#' @export
ggml_new_tensor_1d <- function(ctx, type, ne0) {
  .Call("R_ggml_new_tensor_1d", ctx, as.integer(type), as.numeric(ne0), PACKAGE = "ggmlR")
}

#' Create 2D tensor
#' @export
ggml_new_tensor_2d <- function(ctx, type, ne0, ne1) {
  .Call("R_ggml_new_tensor_2d", ctx, as.integer(type), as.numeric(ne0), as.numeric(ne1), PACKAGE = "ggmlR")
}

#' Set F32 data
#' @export
ggml_set_f32 <- function(tensor, data) {
  invisible(.Call("R_ggml_set_f32", tensor, as.numeric(data), PACKAGE = "ggmlR"))
}

#' Get F32 data
#' @export
ggml_get_f32 <- function(tensor) {
  .Call("R_ggml_get_f32", tensor, PACKAGE = "ggmlR")
}

#' Add tensors
#' @export
ggml_add <- function(ctx, a, b) {
  .Call("R_ggml_add", ctx, a, b, PACKAGE = "ggmlR")
}

#' Multiply tensors
#' @export
ggml_mul <- function(ctx, a, b) {
  .Call("R_ggml_mul", ctx, a, b, PACKAGE = "ggmlR")
}

#' Build forward expand
#' @export
ggml_build_forward_expand <- function(ctx, tensor) {
  .Call("R_ggml_build_forward_expand", ctx, tensor, PACKAGE = "ggmlR")
}

#' Compute graph
#' @export
ggml_graph_compute <- function(ctx, graph) {
  invisible(.Call("R_ggml_graph_compute", ctx, graph, PACKAGE = "ggmlR"))
}

#' Get GGML version
#' @export
ggml_version <- function() {
  .Call("R_ggml_version", PACKAGE = "ggmlR")
}

#' Test GGML
#' @export
ggml_test <- function() {
  .Call("R_ggml_test", PACKAGE = "ggmlR")
}

#' Get number of elements
#' @export
ggml_nelements <- function(tensor) {
  .Call("R_ggml_nelements", tensor, PACKAGE = "ggmlR")
}

#' Get number of bytes
#' @export
ggml_nbytes <- function(tensor) {
  .Call("R_ggml_nbytes", tensor, PACKAGE = "ggmlR")
}

#' Check if GGML is available
#' @export
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
#' ctx <- ggml_init(100 * 1024 * 1024)
#' 
#' # Use context
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000000)
#' 
#' # Reset to reuse memory
#' ggml_reset(ctx)
#' 
#' # Create new tensors in same context
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2000000)
#' 
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
#' # ... do work ...
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
#' Sys.sleep(0.1)
#' elapsed <- ggml_time_ms() - start
#' print(paste("Elapsed:", elapsed, "ms"))
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
#' # ... do fast work ...
#' elapsed <- ggml_time_us() - start
#' print(paste("Elapsed:", elapsed, "us"))
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
ggml_cycles_per_ms <- function() {
  .Call("R_ggml_cycles_per_ms")
}
