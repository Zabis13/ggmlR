# Single-cell adapter: dispatch layer ----------------------------------------
#
# `ggml_run()` is the single entry point between a built task and the engine.
# It validates against the registry, resolves the device (auto -> vulkan when a
# GPU is present, else cpu), and falls back to CPU silently if Vulkan was asked
# for but is unavailable — a missing GPU is never a hard error.

#' Run a single-cell task on the GGML backend
#'
#' Validates a \code{\link{ggml_task}} against \code{\link{ggml_ops_registry}},
#' resolves the compute device, invokes the registered engine and returns a
#' \code{\link{ggml_result}}. This is the dispatch layer: the only place that
#' decides GPU vs CPU. \code{device = "auto"} (the task default) uses Vulkan when
#' a GPU is available and CPU otherwise; \code{device = "vulkan"} on a machine
#' with no GPU degrades to CPU with a message rather than failing.
#'
#' @param task A \code{\link{ggml_task}}.
#' @param backend Optional override of \code{task$device}:
#'   \code{"auto"} (default), \code{"vulkan"} or \code{"cpu"}.
#' @param ... Additional parameters merged over \code{task$params} and passed to
#'   the engine.
#'
#' @return A \code{\link{ggml_result}}.
#' @seealso \code{\link{ggml_task}}, \code{\link{ggml_ops_registry}}
#' @export
ggml_run <- function(task, backend = NULL, ...) {
  if (!inherits(task, "ggml_task"))
    stop("`task` must be a ggml_task (see ggml_task()).", call. = FALSE)

  entry <- ggml_ops_registry(task$op)
  if (is.null(entry))
    stop(sprintf("Unsupported op '%s'. Supported: %s.",
                 task$op, paste(names(ggml_ops_registry()), collapse = ", ")),
         call. = FALSE)

  # merge params: explicit ... override task$params
  params <- utils::modifyList(task$params, list(...))

  # validate required params are present
  missing <- setdiff(entry$params, names(params))
  if (length(missing))
    stop(sprintf("Op '%s' requires parameter(s): %s.",
                 task$op, paste(missing, collapse = ", ")), call. = FALSE)

  # resolve device
  want    <- backend %||% task$device
  backend <- .ggmlr_resolve_backend(want)

  # Densify here unless the op's engine accepts a sparse dgCMatrix directly
  # (sparse_ok). Normalize is sparse_ok: log1p LogNormalize keeps zeros zero, so
  # its engine transforms @x in place and never materialises the dense matrix —
  # that is what lets the full single-cell matrix (tens of GB dense) run at all.
  # Likewise, when chunk_size is set the engine streams the matrix in
  # column-blocks and densifies one block at a time, so leave it sparse here too.
  mat <- task$matrix
  chunked <- !is.null(params$chunk_size) && !is.matrix(mat)
  if (!isTRUE(entry$sparse_ok) && !chunked && !is.matrix(mat)) mat <- as.matrix(mat)

  do.call(entry$engine, c(list(mat = mat, backend = backend), params))
}

# internal: turn a requested device into a concrete, available one
.ggmlr_resolve_backend <- function(want = c("auto", "vulkan", "cpu")) {
  want <- match.arg(want)
  gpu  <- isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE))

  if (want == "cpu")    return("cpu")
  if (want == "auto")   return(if (gpu) "vulkan" else "cpu")
  # want == "vulkan"
  if (gpu) return("vulkan")
  message("ggmlR: Vulkan GPU not available; falling back to CPU.")
  "cpu"
}
