#' @useDynLib ggmlR, .registration = TRUE
#' @keywords internal
"_PACKAGE"

#' Initialize GGML context
#' @param mem_size Memory size in bytes
#' @param no_alloc If TRUE, don't allocate memory for tensors (default: FALSE)
#' @return GGML context pointer
#' @export
#' @examples
#' \donttest{
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
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' ggml_free(ctx)
#' }
ggml_free <- function(ctx) {
  invisible(.Call("R_ggml_free", ctx, PACKAGE = "ggmlR"))
}

#' Get GGML version
#' @return Character string with GGML version
#' @export
#' @examples
#' \donttest{
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
#' \donttest{
#' ggml_test()
#' }
ggml_test <- function() {
  .Call("R_ggml_test", PACKAGE = "ggmlR")
}

#' Check if GGML is available
#' @return TRUE if GGML library is loaded
#' @export
#' @examples
#' \donttest{
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
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' ggml_reset(ctx)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 200)
#' ggml_free(ctx)
#' }
ggml_reset <- function(ctx) {
  invisible(.Call("R_ggml_reset", ctx))
}

#' Timing Utilities
#'
#' Initializes the GGML timing system. Call this once at the beginning
#' of the program before using ggml_time_ms() or ggml_time_us().
#'
#' @return NULL (invisible)
#' @export
#' @examples
#' \donttest{
#' ggml_time_init()
#' start <- ggml_time_ms()
#' Sys.sleep(0.01)
#' elapsed <- ggml_time_ms() - start
#' }
ggml_time_init <- function() {
  invisible(.Call("R_ggml_time_init"))
}

#' @rdname ggml_time_init
#' @details \code{ggml_time_ms()}: Returns the current time in milliseconds since the timer was initialized.
#' Returns: Numeric value representing milliseconds
#' @examples
#' \donttest{
#' ggml_time_init()
#' start <- ggml_time_ms()
#' Sys.sleep(0.01)
#' elapsed <- ggml_time_ms() - start
#' }
#' @export
ggml_time_ms <- function() {
  .Call("R_ggml_time_ms")
}

#' @rdname ggml_time_init
#' @details \code{ggml_time_us()}: Returns the current time in microseconds since the timer was initialized.
#' More precise than ggml_time_ms() for micro-benchmarking.
#' Returns: Numeric value representing microseconds
#' @examples
#' \donttest{
#' ggml_time_init()
#' start <- ggml_time_us()
#' Sys.sleep(0.001)
#' elapsed <- ggml_time_us() - start
#' }
#' @export
ggml_time_us <- function() {
  .Call("R_ggml_time_us")
}

#' @rdname ggml_time_init
#' @details \code{ggml_cycles()}: Returns the current CPU cycle count. Useful for low-level benchmarking.
#' Returns: Numeric value representing CPU cycles
#' @examples
#' \donttest{
#' ggml_cycles()
#' }
#' @export
ggml_cycles <- function() {
  .Call("R_ggml_cycles")
}

#' @rdname ggml_time_init
#' @details \code{ggml_cycles_per_ms()}: Returns an estimate of CPU cycles per millisecond.
#' Useful for converting cycle counts to time.
#' Returns: Numeric value representing cycles per millisecond
#' @examples
#' \donttest{
#' ggml_cycles_per_ms()
#' }
#' @export
ggml_cycles_per_ms <- function() {
  .Call("R_ggml_cycles_per_ms")
}
