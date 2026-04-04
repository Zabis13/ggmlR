# GGUF file reader — low-level access to pre-trained weights

#' Load a GGUF File
#'
#' Opens a GGUF file and reads all metadata and tensor data into memory.
#' Returns an S3 object of class \code{"gguf"} wrapping the internal pointer.
#'
#' @param path Path to a .gguf file.
#' @return An object of class \code{"gguf"}.
#' @export
gguf_load <- function(path) {
  path <- normalizePath(path, mustWork = TRUE)
  ptr <- .Call("R_gguf_load", path)
  info <- .Call("R_gguf_info", ptr)
  structure(list(
    ptr       = ptr,
    path      = path,
    version   = info$version,
    n_tensors = info$n_tensors,
    n_kv      = info$n_kv
  ), class = "gguf")
}

#' @export
print.gguf <- function(x, ...) {
  cat(sprintf("GGUF file: %s\n", basename(x$path)))
  cat(sprintf("  Version:  %d\n", x$version))
  cat(sprintf("  Tensors:  %d\n", x$n_tensors))
  cat(sprintf("  Metadata: %d key-value pairs\n", x$n_kv))
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
  .Call("R_gguf_tensor_data", x$ptr, as.character(name))
}

#' Free GGUF Resources
#'
#' Explicitly frees the internal GGUF context. Called automatically by the
#' garbage collector, but can be called manually to release memory sooner.
#'
#' @param x A \code{gguf} object.
#' @export
gguf_free <- function(x) {
  if (!inherits(x, "gguf")) stop("Expected a gguf object")
  .Call("R_gguf_free", x$ptr)
  invisible(NULL)
}
