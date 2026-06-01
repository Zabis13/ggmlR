# GGUF file reader — low-level access to pre-trained weights

#' Load a GGUF File
#'
#' Opens a GGUF file and reads its metadata. By default also reads tensor data
#' into memory; with \code{meta_only = TRUE} only the header and key-value
#' metadata are read (no tensor data is allocated), which is cheap and enough
#' for inspecting architecture / type fields. Returns an S3 object of class
#' \code{"gguf"} wrapping the internal pointer.
#'
#' @param path Path to a .gguf file.
#' @param meta_only If \code{TRUE}, read only the header and metadata without
#'   allocating tensor data. Metadata, tensor names and tensor info remain
#'   available; \code{\link{gguf_tensor_data}} is not (reload with
#'   \code{meta_only = FALSE}). Default \code{FALSE}.
#' @return An object of class \code{"gguf"}.
#' @export
gguf_load <- function(path, meta_only = FALSE) {
  path <- normalizePath(path, mustWork = TRUE)
  ptr <- .Call("R_gguf_load", path, as.logical(meta_only))
  info <- .Call("R_gguf_info", ptr)
  structure(list(
    ptr       = ptr,
    path      = path,
    version   = info$version,
    n_tensors = info$n_tensors,
    n_kv      = info$n_kv,
    meta_only = isTRUE(meta_only)
  ), class = "gguf")
}

#' @export
print.gguf <- function(x, ...) {
  cat(sprintf("GGUF file: %s\n", basename(x$path)))
  cat(sprintf("  Version:  %d\n", x$version))
  cat(sprintf("  Tensors:  %d\n", x$n_tensors))
  cat(sprintf("  Metadata: %d key-value pairs\n", x$n_kv))
  if (isTRUE(x$meta_only)) cat("  (meta_only: tensor data not loaded)\n")
  invisible(x)
}

#' Get GGUF Metadata
#'
#' Returns all key-value metadata pairs from a GGUF file as a named list.
#'
#' @param x A \code{gguf} object from \code{\link{gguf_load}}.
#' @return A named list of metadata values.
#' @export
gguf_metadata <- function(x) {
  if (!inherits(x, "gguf")) stop("Expected a gguf object")
  .Call("R_gguf_metadata", x$ptr)
}

#' List Tensor Names in a GGUF File
#'
#' @param x A \code{gguf} object.
#' @return Character vector of tensor names.
#' @export
gguf_tensor_names <- function(x) {
  if (!inherits(x, "gguf")) stop("Expected a gguf object")
  .Call("R_gguf_tensor_names", x$ptr)
}

#' Get Tensor Info
#'
#' Returns name, shape, type, and size in bytes for a single tensor.
#'
#' When the file was opened with \code{meta_only = TRUE}, the per-dimension
#' \code{shape} is \code{NA} (the public GGUF API does not expose tensor
#' dimensions without allocating tensors); \code{name}, \code{type} and
#' \code{size_bytes} are still returned.
#'
#' @param x A \code{gguf} object.
#' @param name Tensor name (character).
#' @return A list with elements \code{name}, \code{shape}, \code{type},
#'   \code{size_bytes}.
#' @export
gguf_tensor_info <- function(x, name) {
  if (!inherits(x, "gguf")) stop("Expected a gguf object")
  .Call("R_gguf_tensor_info", x$ptr, as.character(name))
}

#' Extract Tensor Data
#'
#' Dequantizes (if needed) and returns tensor weights as an R numeric array
#' with dimensions matching the tensor shape.
#'
#' @param x A \code{gguf} object.
#' @param name Tensor name (character).
#' @return A numeric array.
#' @export
gguf_tensor_data <- function(x, name) {
  if (!inherits(x, "gguf")) stop("Expected a gguf object")
  if (isTRUE(x$meta_only)) {
    stop("GGUF loaded with meta_only=TRUE, tensor data not available. ",
         "Reload with gguf_load(path, meta_only=FALSE)", call. = FALSE)
  }
  .Call("R_gguf_tensor_data", x$ptr, as.character(name))
}

#' Free GGUF Resources
#'
#' Explicitly frees the internal GGUF context. Called automatically by the
#' garbage collector, but can be called manually to release memory sooner.
#'
#' @param x A \code{gguf} object.
#' @return Called for its side effect (releases the GGUF context); invisibly returns \code{NULL}.
#' @export
gguf_free <- function(x) {
  if (!inherits(x, "gguf")) stop("Expected a gguf object")
  .Call("R_gguf_free", x$ptr)
  invisible(NULL)
}
