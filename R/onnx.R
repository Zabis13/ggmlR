# onnx.R — ONNX model inference via ggml backend
#
# Minimal API:
#   onnx_load(path, device, input_shapes)  — load .onnx file, build ggml graph
#   onnx_summary(model)      — model metadata
#   onnx_run(model, inputs)  — run inference
#   onnx_inputs(model)       — list expected inputs and shapes

#' Load an ONNX model
#'
#' Parses an .onnx file, builds a ggml computation graph, and allocates
#' tensors on the specified device. Weights are loaded via memory-mapped
#' file (zero-copy where possible).
#'
#' @param path Path to .onnx file.
#' @param device Backend device: \code{"vulkan"} (default if available)
#'   or \code{"cpu"}.
#' @param input_shapes Optional named list of integer vectors specifying
#'   fixed shapes for inputs with dynamic dimensions. Names must match
#'   input tensor names. Each shape must include all dimensions including
#'   batch, e.g. \code{list(image = c(1L, 3L, 224L, 224L))}.
#'   Required when the model has dynamic dimensions and no default shape.
#' @return An opaque model object (external pointer) for use with
#'   \code{onnx_run()}, \code{onnx_summary()}, and \code{onnx_inputs()}.
#' @export
onnx_load <- function(path, device = NULL, input_shapes = NULL) {
  path <- normalizePath(path, mustWork = TRUE)

  # Parse the ONNX protobuf
  onnx_ptr <- .Call("R_onnx_load", path)

  # Get summary before building (onnx_ptr gets consumed by build)
  info <- .Call("R_onnx_summary", onnx_ptr)

  # Override input shapes if provided
  if (!is.null(input_shapes)) {
    stopifnot(is.list(input_shapes), !is.null(names(input_shapes)))
    shape_names <- names(input_shapes)
    shape_vals <- lapply(input_shapes, as.integer)
    .Call("R_onnx_override_input_shapes", onnx_ptr, shape_names, shape_vals)
  }

  # Build ggml graph + allocate on device
  ctx_ptr <- .Call("R_onnx_build", onnx_ptr, device)

  # Check for remaining dynamic dimensions
  inp <- .Call("R_onnx_inputs", ctx_ptr)
  for (nm in names(inp)) {
    if (any(inp[[nm]] < 0L)) {
      dims_str <- paste(ifelse(inp[[nm]] < 0L, "?", inp[[nm]]), collapse = "x")
      stop("Input '", nm, "' has dynamic shape [", dims_str, "]. ",
           "Specify fixed shape via input_shapes parameter, e.g. ",
           "onnx_load(\"", basename(path), "\", input_shapes = list(",
           nm, " = c(1, 3, 224, 224))). ",
           "Alternatively, re-export the model with static shapes.",
           call. = FALSE)
    }
  }

  structure(
    list(
      ptr          = ctx_ptr,
      path         = path,
      ir_version   = info$ir_version,
      opset        = info$opset_version,
      producer     = info$producer,
      graph_name   = info$graph_name,
      n_nodes      = info$n_nodes,
      n_weights    = info$n_initializers,
      ops          = info$ops
    ),
    class = "onnx_model"
  )
}

#' Print ONNX model summary
#'
#' @param x An \code{onnx_model} object.
#' @param ... Ignored.
#' @return Invisibly returns \code{x}.
#' @export
print.onnx_model <- function(x, ...) {
  cat("ONNX Model:", x$graph_name, "\n")
  cat("  Producer:", x$producer, "\n")
  cat("  IR version:", x$ir_version, "/ Opset:", x$opset, "\n")
  cat("  Nodes:", x$n_nodes, "/ Weights:", x$n_weights, "\n")
  cat("  Ops:", paste(x$ops, collapse = ", "), "\n")
  invisible(x)
}

#' ONNX model summary
#'
#' Returns metadata about a loaded ONNX model.
#'
#' @param model An \code{onnx_model} object from \code{onnx_load()}.
#' @return A list with \code{ir_version}, \code{opset_version},
#'   \code{producer}, \code{graph_name}, \code{n_nodes},
#'   \code{n_initializers}, and \code{ops}.
#' @export
onnx_summary <- function(model) {
  stopifnot(inherits(model, "onnx_model"))
  list(
    ir_version     = model$ir_version,
    opset_version  = model$opset,
    producer       = model$producer,
    graph_name     = model$graph_name,
    n_nodes        = model$n_nodes,
    n_initializers = model$n_weights,
    ops            = model$ops
  )
}

#' Run ONNX model inference
#'
#' @param model An \code{onnx_model} object from \code{onnx_load()}.
#' @param inputs A named list of numeric vectors/matrices.
#'   Names must match the model's input tensor names.
#'   Use \code{onnx_inputs()} to see expected names and shapes.
#' @return A named list of output tensors (numeric vectors with dim
#'   attributes for multi-dimensional outputs).
#' @export
onnx_run <- function(model, inputs) {
  stopifnot(inherits(model, "onnx_model"))
  stopifnot(is.list(inputs), !is.null(names(inputs)))

  input_names <- names(inputs)
  input_data <- lapply(inputs, function(x) as.numeric(x))

  .Call("R_onnx_run", model$ptr, input_names, input_data)
}

#' List ONNX model inputs
#'
#' Returns the names and shapes of model inputs (excluding weight
#' initializers). Use this to know what to pass to \code{onnx_run()}.
#'
#' @param model An \code{onnx_model} object from \code{onnx_load()}.
#' @return A named list where names are input tensor names and values
#'   are integer vectors of dimension sizes (-1 for dynamic dimensions).
#' @export
onnx_inputs <- function(model) {
  stopifnot(inherits(model, "onnx_model"))
  .Call("R_onnx_inputs", model$ptr)
}
