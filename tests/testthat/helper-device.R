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

# Skip a test that needs a real GPU.
#
# The obvious-looking `ggml_backend_dev_count() < 1` does NOT work for this: it
# counts every backend device, and the CPU one is always present, so the
# condition is false on a CPU-only machine and the test runs anyway. Worse,
# ag_device("gpu") does not fail there either -- ggml_backend_init_best() falls
# back to the CPU backend and returns it -- so the test proceeds down the GPU
# path with a CPU backend under it and segfaults. That is what took down R CMD
# check on r-hub (test-ag-grad-underflow.R:62, no traceback, since the crash is
# in C inside .Call).
#
# Ask about the GPU itself instead, the way test-vulkan.R already does.
skip_if_no_gpu <- function(what = "No GPU available") {
  ok <- tryCatch(
    isTRUE(ggml_vulkan_available()) && ggml_vulkan_device_count() > 0,
    error = function(e) FALSE
  )
  skip_if_not(ok, what)
}

# read the current device without assuming the internals are exported
.ag_device_state_device <- function() {
  st <- get(".ag_device_state", envir = asNamespace("ggmlR"))
  st$device %||% "cpu"
}

# `%||%` is internal to ggmlR; define a local copy for the helpers above
if (!exists("%||%")) `%||%` <- function(a, b) if (!is.null(a)) a else b
