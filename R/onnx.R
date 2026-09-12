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
#' @param n_threads Number of CPU threads. \code{NULL} (default) reads
#'   \code{getOption("ggmlR.n_threads")}; if that is also unset, uses
#'   \code{parallel::detectCores() - 1} (minimum 1).
#' @param dtype Weight precision: \code{"f32"} (default) or \code{"f16"}.
#'   When \code{"f16"}, large weight tensors (>= 256 elements) are stored
#'   in half-precision for faster Vulkan compute and lower VRAM usage.
#'   Small tensors (bias, scalars, batch-norm params) remain in F32
#'   for numerical stability. Inputs and outputs are always F32.
#' @return An opaque model object (external pointer) for use with
#'   \code{onnx_run()}, \code{onnx_summary()}, and \code{onnx_inputs()}.
#' @export
onnx_load <- function(path, device = NULL, input_shapes = NULL, n_threads = NULL,
                      dtype = "f32") {
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

  # Resolve n_threads: argument > auto (all cores minus 1)
  if (is.null(n_threads)) {
    nc <- parallel::detectCores(logical = FALSE)
    if (is.na(nc)) nc <- 1L
    n_threads <- max(nc - 1L, 1L)
  }
  n_threads <- as.integer(n_threads)

  # Validate dtype
  dtype <- match.arg(dtype, c("f32", "f16", "fp16", "float16"))
  if (dtype %in% c("fp16", "float16")) dtype <- "f16"

  # Build ggml graph + allocate on device
  ctx_ptr <- .Call("R_onnx_build", onnx_ptr, device, n_threads, dtype)

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
      ops          = info$ops,
      dtype        = dtype
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
  if (!is.null(x$dtype) && x$dtype != "f32")
    cat("  Weight dtype:", toupper(x$dtype), "\n")
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
#' @return A named list of output tensors: numeric vectors carrying a
#'   \code{dim} attribute when the output has more than one dimension.
#'
#'   The buffers are row-major, as ONNX stores them, so \code{dim} is given in
#'   storage order -- fastest-varying axis first -- which is the reverse of the
#'   shape the ONNX file declares. An output declared \code{[nbox, 4]} arrives
#'   as \code{dim = c(4, nbox)}, one detection per column. This keeps \code{dim}
#'   an honest description of the memory, so ordinary R indexing is correct;
#'   use \code{\link{predict.onnx_model}} to get arrays in the declared ONNX
#'   order instead.
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

#' ONNX model device/scheduler diagnostics
#'
#' Returns information about backend placement: which backends are
#' available, how the scheduler splits the graph, and how many ops
#' are supported by GPU vs CPU-only.
#'
#' @param model An \code{onnx_model} object from \code{onnx_load()}.
#' @return A list with:
#'   \describe{
#'     \item{backends}{Character vector of backend names (e.g. \code{"Vulkan0"}, \code{"CPU"})}
#'     \item{n_backends}{Number of backends}
#'     \item{n_splits}{Number of scheduler splits (1 = all on one backend)}
#'     \item{n_nodes}{Total graph nodes}
#'     \item{gpu_ops}{Ops supported by GPU backend}
#'     \item{cpu_ops}{Ops that can only run on CPU}
#'     \item{cpu_only_ops}{Named integer vector: op type => count (empty if all on GPU)}
#'   }
#' @export
onnx_device_info <- function(model) {
  stopifnot(inherits(model, "onnx_model"))
  .Call("R_onnx_device_info", model$ptr)
}

#' ONNX graph structure
#'
#' The parsed graph as a table: one row per node, with its operator type and
#' the names of its inputs and outputs. Use it to find which node produces a
#' given edge, or what an edge feeds into, without re-parsing the file.
#'
#' Reading the structure beats searching the \code{.onnx} binary for a name:
#' a name occurs both as a producer's output and as its consumers' input, and
#' telling those apart by hand means getting the protobuf field numbers right
#' every time.
#'
#' @param model An \code{onnx_model} object from \code{onnx_load()}.
#' @param producer_of Optional edge name. When given, returns the rows whose
#'   node lists that name among its outputs.
#' @param consumer_of Optional edge name. When given, returns the rows whose
#'   node lists that name among its inputs.
#' @return A data frame with columns \code{index}, \code{op_type},
#'   \code{output} (the first output) and list-columns \code{outputs} and
#'   \code{inputs}.
#' @export
#' @examples
#' \donttest{
#' if (file.exists("model.onnx")) {
#'   m <- onnx_load("model.onnx")
#'   g <- onnx_graph_info(m)
#'   head(g)
#'   onnx_graph_info(m, producer_of = "881")
#' }
#' }
onnx_graph_info <- function(model, producer_of = NULL, consumer_of = NULL) {
  stopifnot(inherits(model, "onnx_model"))
  g <- .Call("R_onnx_graph_info", model$ptr)
  df <- data.frame(index = g$index, op_type = g$op_type, output = g$output,
                   stringsAsFactors = FALSE)
  df$outputs <- g$outputs
  df$inputs  <- g$inputs
  if (!is.null(producer_of))
    df <- df[vapply(df$outputs, function(o) producer_of %in% o, logical(1)), ]
  if (!is.null(consumer_of))
    df <- df[vapply(df$inputs, function(o) consumer_of %in% o, logical(1)), ]
  df
}

# ============================================================================
# predict() -- keras-compatible entry point for ONNX models
# ============================================================================

# Split an array along its first (sample) axis, returning rows start:end.
# ONNX arrays are row-major with the batch dimension first, matching R's
# convention for the leading index but not its column-major storage, so the
# slice is expressed with indices rather than raw offsets.
onnx_slice_samples <- function(a, idx) {
  d <- dim(a)
  if (is.null(d) || length(d) == 1L) return(a[idx])
  args <- rep(list(quote(expr = )), length(d))
  args[[1L]] <- idx
  do.call(`[`, c(list(a), args, list(drop = FALSE)))
}

# R arrays are column-major, ONNX buffers are row-major with the batch
# dimension first.  onnx_run() takes and returns buffers in that row-major
# order, so the two directions need an explicit reversal of the axes -- the same
# thing the tests do by hand with as.vector(t(x)) in the 2D case.
onnx_flatten_rowmajor <- function(a) {
  d <- dim(a)
  if (is.null(d) || length(d) == 1L) return(as.vector(a))
  as.vector(aperm(a, rev(seq_along(d))))
}

# Inverse of onnx_flatten_rowmajor(): read a row-major buffer into an R array
# with the given ONNX dimensions.
onnx_array_rowmajor <- function(v, dims) {
  if (length(dims) <= 1L) return(as.vector(v))
  aperm(array(as.vector(v), dim = rev(dims)), rev(seq_along(dims)))
}

# Bind a list of same-shaped arrays along the first axis.
onnx_bind_samples <- function(parts) {
  if (length(parts) == 1L) return(parts[[1L]])
  d <- dim(parts[[1L]])
  if (is.null(d) || length(d) <= 2L) return(do.call(rbind, parts))

  # >2D: concatenating along the first axis means interleaving in storage
  # order, so go through a permutation that puts the sample axis last.
  perm     <- c(seq.int(2L, length(d)), 1L)
  permuted <- lapply(parts, function(p) aperm(p, perm))
  out      <- array(unlist(permuted, use.names = FALSE),
                    dim = c(d[-1L], sum(vapply(parts, function(p) dim(p)[1L], 1L))))
  aperm(out, order(perm))
}

#' Predict with an ONNX Model
#'
#' Runs inference over \code{x}, batching it to the fixed input shape the
#' model was loaded with.  This is the keras-compatible entry point: together
#' with the \code{ggml_sequential_model} and \code{ggml_functional_model}
#' methods it gives ggmlR and ONNX models a single \code{predict()} interface.
#'
#' Unlike \code{\link{onnx_run}}, which takes a named list of inputs and always
#' returns a named list of outputs, this method takes an array (or a named list
#' for multi-input models) and unwraps a single output.  The output keeps the
#' dimensions the model produced: a 2D output comes back as a matrix, a 4D
#' output stays 4D.
#'
#' An ONNX graph is built for one fixed batch size at \code{\link{onnx_load}}
#' time and cannot be resized afterwards, so \code{x} is split into chunks of
#' that size.  A trailing partial chunk is padded to a full batch and the
#' padding is dropped from the result.
#'
#' @param object An \code{onnx_model} from \code{\link{onnx_load}}.
#' @param x Input data: an array or matrix whose first dimension indexes
#'   samples, or a named list of such arrays for multi-input models.
#' @param batch_size Ignored, with a warning if not \code{NULL}.  The batch
#'   size is fixed when the model is loaded; pass \code{input_shapes} to
#'   \code{\link{onnx_load}} to change it.  The argument exists so the
#'   signature matches the other \code{predict()} methods.
#' @param ... Ignored.
#' @return For a single-output model, an array with the model's output
#'   dimensions and \code{nrow(x)} samples in the first one (a matrix when the
#'   output is 2D).  For a multi-output model, a named list of such arrays.
#' @seealso \code{\link{onnx_run}} for the lower-level named-list interface.
#' @export
predict.onnx_model <- function(object, x, batch_size = NULL, ...) {
  stopifnot(inherits(object, "onnx_model"))
  if (!is.null(batch_size)) {
    warning("ONNX models have a fixed batch size chosen at onnx_load(); ",
            "ignoring batch_size. Use onnx_load(input_shapes = ...) instead.",
            call. = FALSE)
  }

  spec     <- onnx_inputs(object)
  in_names <- names(spec)

  # Normalise x to a named list of arrays, one per model input.
  if (!is.list(x)) {
    if (length(in_names) != 1L) {
      stop("Model has ", length(in_names), " inputs (",
           paste(in_names, collapse = ", "),
           "); pass a named list, not a single array.", call. = FALSE)
    }
    x <- stats::setNames(list(x), in_names)
  } else {
    if (is.null(names(x))) {
      stop("Input list must be named. Expected: ",
           paste(in_names, collapse = ", "), call. = FALSE)
    }
    missing_in <- setdiff(in_names, names(x))
    if (length(missing_in) > 0L) {
      stop("Missing input(s): ", paste(missing_in, collapse = ", "),
           call. = FALSE)
    }
    x <- x[in_names]
  }

  # The batch each input was built for, and the per-sample shape after it.
  batch_of  <- vapply(spec, function(s) as.integer(s[1L]), 1L)
  model_bs  <- batch_of[[1L]]
  if (any(batch_of != model_bs)) {
    stop("Inputs disagree on batch size (",
         paste(sprintf("%s=%d", in_names, batch_of), collapse = ", "),
         "); cannot batch automatically. Use onnx_run() directly.",
         call. = FALSE)
  }

  # Sample count, and a per-input view whose first dimension is the batch.
  as_batched <- function(a, s) {
    per_sample <- s[-1L]
    if (is.null(dim(a))) {
      # A bare vector is one sample when it matches the per-sample size,
      # otherwise a stack of them.
      n_per <- max(prod(per_sample), 1L)
      dim(a) <- c(length(a) %/% n_per, per_sample)
    }
    a
  }
  x <- Map(as_batched, x, spec)

  n_samples <- unique(vapply(x, function(a) dim(a)[1L], 1L))
  if (length(n_samples) != 1L) {
    stop("Inputs disagree on sample count: ",
         paste(vapply(x, function(a) dim(a)[1L], 1L), collapse = ", "),
         call. = FALSE)
  }
  if (n_samples == 0L) stop("No samples provided.", call. = FALSE)

  starts <- seq.int(1L, n_samples, by = model_bs)
  chunks <- vector("list", length(starts))

  for (k in seq_along(starts)) {
    from <- starts[[k]]
    to   <- min(from + model_bs - 1L, n_samples)
    idx  <- seq.int(from, to)
    pad  <- model_bs - length(idx)
    # Repeat the last sample to fill a short final batch; the padded rows are
    # discarded below, so the values only need to be valid, not meaningful.
    if (pad > 0L) idx <- c(idx, rep(idx[length(idx)], pad))

    batch_in <- lapply(x, function(a) onnx_flatten_rowmajor(onnx_slice_samples(a, idx)))
    res      <- onnx_run(object, batch_in)
    # onnx_run() reports dim() in storage order (fastest axis first); the ONNX
    # shape is its reverse.  predict() returns ordinary R arrays, so undo the
    # row-major layout here.
    res <- lapply(res, function(o) {
      d <- dim(o)
      if (is.null(d)) o else onnx_array_rowmajor(o, rev(d))
    })
    if (pad > 0L) {
      res <- lapply(res, function(o) onnx_slice_samples(o, seq_len(model_bs - pad)))
    }
    chunks[[k]] <- res
  }

  out_names <- names(chunks[[1L]])
  out <- lapply(out_names, function(nm) {
    onnx_bind_samples(lapply(chunks, function(cc) cc[[nm]]))
  })
  names(out) <- out_names

  # Single output: hand back the array itself rather than a one-element list.
  if (length(out) == 1L) out[[1L]] else out
}
