# Logging and debugging functions

# ============================================================================
# Logging Functions
# ============================================================================

#' Logging and Abort Callbacks
#'
#' Redirects GGML log messages to R's message system:
#' - INFO/DEBUG messages go to stdout (via Rprintf)
#' - WARN/ERROR messages go to stderr (via REprintf)
#'
#' @return NULL invisibly
#' @export
#' @family logging
#' @examples
#' \donttest{
#' ggml_log_set_r()
#' # Now GGML messages will appear in R console
#' }
ggml_log_set_r <- function() {
  invisible(.Call("R_ggml_log_set_r"))
}

#' @rdname ggml_log_set_r
#' @details \code{ggml_log_set_default()}: Restores GGML to default logging behavior (stderr output).
#' Returns: NULL invisibly
#' @export
ggml_log_set_default <- function() {
  invisible(.Call("R_ggml_log_set_default"))
}

#' @rdname ggml_log_set_r
#' @details \code{ggml_log_is_r_enabled()}: 
#' Returns: Logical indicating if R-compatible logging is active
#' @export
ggml_log_is_r_enabled <- function() {
  .Call("R_ggml_log_is_r_enabled")
}

# ============================================================================
# Abort Callback Functions
# ============================================================================

#' @rdname ggml_log_set_r
#' @details \code{ggml_set_abort_callback_r()}: Converts GGML abort calls into R errors (via Rf_error).
#' This allows R to catch GGML failures with tryCatch.
#' Returns: NULL invisibly
#' @examples
#' \donttest{
#' ggml_set_abort_callback_r()
#' # Now GGML aborts will become R errors
#' result <- tryCatch({
#'   # ... ggml operations that might fail ...
#' }, error = function(e) {
#'   message("GGML error caught: ", e$message)
#' })
#' }
#' @export
ggml_set_abort_callback_r <- function() {
  invisible(.Call("R_ggml_set_abort_callback_r"))
}

#' @rdname ggml_log_set_r
#' @details \code{ggml_set_abort_callback_default()}: Restores GGML to default abort behavior (prints to stderr and aborts).
#' Returns: NULL invisibly
#' @export
ggml_set_abort_callback_default <- function() {
  invisible(.Call("R_ggml_set_abort_callback_default"))
}

#' @rdname ggml_log_set_r
#' @details \code{ggml_abort_is_r_enabled()}: 
#' Returns: Logical indicating if R-compatible abort handling is active
#' @export
ggml_abort_is_r_enabled <- function() {
  .Call("R_ggml_abort_is_r_enabled")
}

# ============================================================================
# Op Params Functions
# ============================================================================

#' Operation Parameters
#'
#' Returns the raw op_params bytes from a tensor. These parameters
#' control operation-specific behavior (e.g., precision, mode).
#'
#' @param tensor External pointer to tensor
#' @return Raw vector of op_params bytes
#' @export
#' @family tensor
ggml_get_op_params <- function(tensor) {
  .Call("R_ggml_get_op_params", tensor)
}

#' @rdname ggml_get_op_params
#' @details \code{ggml_set_op_params()}: Sets the raw op_params bytes for a tensor.
#' Returns: NULL invisibly
#' @param params Raw vector of parameters (max 64 bytes)
#' @export
ggml_set_op_params <- function(tensor, params) {
  if (!is.raw(params)) {
    params <- as.raw(params)
  }
  invisible(.Call("R_ggml_set_op_params", tensor, params))
}

#' @rdname ggml_get_op_params
#' @details \code{ggml_get_op_params_i32()}: Gets a single int32 value from tensor op_params at given index.
#' Returns: Integer value
#' @export
ggml_get_op_params_i32 <- function(tensor, index) {
  .Call("R_ggml_get_op_params_i32", tensor, as.integer(index))
}

#' @rdname ggml_get_op_params
#' @details \code{ggml_set_op_params_i32()}: Sets a single int32 value in tensor op_params at given index.
#' Returns: NULL invisibly
#' @export
ggml_set_op_params_i32 <- function(tensor, index, value) {
  invisible(.Call("R_ggml_set_op_params_i32", tensor, as.integer(index),
                  as.integer(value)))
}

#' @rdname ggml_get_op_params
#' @details \code{ggml_get_op_params_f32()}: Gets a single float value from tensor op_params at given index.
#' Returns: Numeric value
#' @param index 0-based index (0-15 for 64-byte op_params)
#' @export
ggml_get_op_params_f32 <- function(tensor, index) {
  .Call("R_ggml_get_op_params_f32", tensor, as.integer(index))
}

#' @rdname ggml_get_op_params
#' @details \code{ggml_set_op_params_f32()}: Sets a single float value in tensor op_params at given index.
#' Returns: NULL invisibly
#' @param value Numeric value to set
#' @export
ggml_set_op_params_f32 <- function(tensor, index, value) {
  invisible(.Call("R_ggml_set_op_params_f32", tensor, as.integer(index),
                  as.numeric(value)))
}
