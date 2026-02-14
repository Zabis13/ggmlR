# Package-level state
.ggmlr_state <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Redirect GGML log messages through R's logging system.
  # The R callback suppresses DEBUG-level messages (scheduler realloc,
  # graph allocation internals) while forwarding INFO/WARN/ERROR.
  ggml_log_set_r()
  ggml_set_abort_callback_r()

  # Track whether backend message has been shown
  .ggmlr_state$backend_msg_shown <- FALSE
}
