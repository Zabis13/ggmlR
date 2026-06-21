# The single-cell engines (PCA / UMAP / neighbours) call ag_device("gpu")
# internally, which flips the package-global device state in .ag_device_state and
# leaves it on "gpu". That state leaks across test files: a later test that does
# not set a device inherits "gpu" and may run autograd training on the GPU, where
# f16 accumulation can diverge to NaN on un-scaled data. Tests that run those
# engines call local_cpu_device() so the device is restored to "cpu" when the
# calling frame (a test_that block or a whole file) exits.
local_cpu_device <- function(.env = parent.frame()) {
  prev <- tryCatch(.ag_device_state_device(), error = function(e) "cpu")
  withr::defer(ag_device("cpu"), envir = .env)
  invisible(prev)
}

# read the current device without assuming the internals are exported
.ag_device_state_device <- function() {
  st <- get(".ag_device_state", envir = asNamespace("ggmlR"))
  st$device %||% "cpu"
}

# `%||%` is internal to ggmlR; define a local copy for the helpers above
if (!exists("%||%")) `%||%` <- function(a, b) if (!is.null(a)) a else b
