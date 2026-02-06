# Logging and debugging functions

# ============================================================================
# Logging Functions
# ============================================================================

#' Enable R-compatible GGML Logging
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

#' Restore Default GGML Logging
#'
#' Restores GGML to default logging behavior (stderr output).
#'
#' @return NULL invisibly
#' @export
#' @family logging
ggml_log_set_default <- function() {
  invisible(.Call("R_ggml_log_set_default"))
}

#' Check if R Logging is Enabled
#'
#' @return Logical indicating if R-compatible logging is active
#' @export
#' @family logging
ggml_log_is_r_enabled <- function() {
  .Call("R_ggml_log_is_r_enabled")
}

# ============================================================================
# Abort Callback Functions
# ============================================================================

#' Enable R-compatible Abort Handling
#'
#' Converts GGML abort calls into R errors (via Rf_error).
#' This allows R to catch GGML failures with tryCatch.
#'
#' @return NULL invisibly
#' @export
#' @family logging
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
ggml_set_abort_callback_r <- function() {
  invisible(.Call("R_ggml_set_abort_callback_r"))
}

#' Restore Default Abort Behavior
#'
#' Restores GGML to default abort behavior (prints to stderr and aborts).
#'
#' @return NULL invisibly
#' @export
#' @family logging
ggml_set_abort_callback_default <- function() {
  invisible(.Call("R_ggml_set_abort_callback_default"))
}

#' Check if R Abort Handler is Enabled
#'
#' @return Logical indicating if R-compatible abort handling is active
#' @export
#' @family logging
ggml_abort_is_r_enabled <- function() {
  .Call("R_ggml_abort_is_r_enabled")
}

# ============================================================================
# Op Params Functions
# ============================================================================

#' Get Tensor Operation Parameters
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

#' Set Tensor Operation Parameters
#'
#' Sets the raw op_params bytes for a tensor.
#'
#' @param tensor External pointer to tensor
#' @param params Raw vector of parameters (max 64 bytes)
#' @return NULL invisibly
#' @export
#' @family tensor
ggml_set_op_params <- function(tensor, params) {
  if (!is.raw(params)) {
    params <- as.raw(params)
  }
  invisible(.Call("R_ggml_set_op_params", tensor, params))
}

#' Get Integer Op Parameter
#'
#' Gets a single int32 value from tensor op_params at given index.
#'
#' @param tensor External pointer to tensor
#' @param index 0-based index (0-15 for 64-byte op_params)
#' @return Integer value
#' @export
#' @family tensor
ggml_get_op_params_i32 <- function(tensor, index) {
  .Call("R_ggml_get_op_params_i32", tensor, as.integer(index))
}

#' Set Integer Op Parameter
#'
#' Sets a single int32 value in tensor op_params at given index.
#'
#' @param tensor External pointer to tensor
#' @param index 0-based index (0-15 for 64-byte op_params)
#' @param value Integer value to set
#' @return NULL invisibly
#' @export
#' @family tensor
ggml_set_op_params_i32 <- function(tensor, index, value) {
  invisible(.Call("R_ggml_set_op_params_i32", tensor, as.integer(index),
                  as.integer(value)))
}

#' Get Float Op Parameter
#'
#' Gets a single float value from tensor op_params at given index.
#'
#' @param tensor External pointer to tensor
#' @param index 0-based index (0-15 for 64-byte op_params)
#' @return Numeric value
#' @export
#' @family tensor
ggml_get_op_params_f32 <- function(tensor, index) {
  .Call("R_ggml_get_op_params_f32", tensor, as.integer(index))
}

#' Set Float Op Parameter
#'
#' Sets a single float value in tensor op_params at given index.
#'
#' @param tensor External pointer to tensor
#' @param index 0-based index (0-15 for 64-byte op_params)
#' @param value Numeric value to set
#' @return NULL invisibly
#' @export
#' @family tensor
ggml_set_op_params_f32 <- function(tensor, index, value) {
  invisible(.Call("R_ggml_set_op_params_f32", tensor, as.integer(index),
                  as.numeric(value)))
}
