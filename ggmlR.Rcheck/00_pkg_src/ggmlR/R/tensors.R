#' GGML Data Types
#'
#' Constants representing different data types supported by GGML.
#'
#' @format Integer constants
#' @details
#' \itemize{
#'   \item \code{GGML_TYPE_F32}: 32-bit floating point (default)
#'   \item \code{GGML_TYPE_F16}: 16-bit floating point (half precision)
#'   \item \code{GGML_TYPE_Q4_0}: 4-bit quantization type 0
#'   \item \code{GGML_TYPE_Q4_1}: 4-bit quantization type 1
#'   \item \code{GGML_TYPE_Q8_0}: 8-bit quantization type 0
#'   \item \code{GGML_TYPE_I32}: 32-bit integer
#' }
#' @export
#' @examples
#' \dontrun{
#' GGML_TYPE_F32
#' GGML_TYPE_F16
#' GGML_TYPE_I32
#' }
GGML_TYPE_F32  <- 0L

#' @rdname GGML_TYPE_F32
#' @export
GGML_TYPE_F16  <- 1L

#' @rdname GGML_TYPE_F32
#' @export
GGML_TYPE_Q4_0 <- 2L

#' @rdname GGML_TYPE_F32
#' @export
GGML_TYPE_Q4_1 <- 3L

#' @rdname GGML_TYPE_F32
#' @export
GGML_TYPE_Q8_0 <- 8L

#' @rdname GGML_TYPE_F32
#' @export
GGML_TYPE_I32  <- 26L

#' Create 1D Tensor
#' @param ctx GGML context
#' @param type Data type
#' @param ne0 Size
#' @return Tensor pointer
#' @export
ggml_new_tensor_1d <- function(ctx, type = GGML_TYPE_F32, ne0) {
  .Call("R_ggml_new_tensor_1d", ctx, as.integer(type), as.numeric(ne0))
}

#' Create 2D Tensor
#' @param ctx GGML context
#' @param type Data type
#' @param ne0 Rows
#' @param ne1 Columns
#' @return Tensor pointer
#' @export
ggml_new_tensor_2d <- function(ctx, type = GGML_TYPE_F32, ne0, ne1) {
  .Call("R_ggml_new_tensor_2d", ctx, as.integer(type),
        as.numeric(ne0), as.numeric(ne1))
}

#' Create 3D Tensor
#' @param ctx GGML context
#' @param type Data type (default GGML_TYPE_F32)
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @param ne2 Size of dimension 2
#' @return Tensor pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 10, 20, 30)
#' ggml_nelements(t)
#' ggml_free(ctx)
#' }
ggml_new_tensor_3d <- function(ctx, type = GGML_TYPE_F32, ne0, ne1, ne2) {
  .Call("R_ggml_new_tensor_3d", ctx, as.integer(type),
        as.numeric(ne0), as.numeric(ne1), as.numeric(ne2))
}

#' Create 4D Tensor
#' @param ctx GGML context
#' @param type Data type (default GGML_TYPE_F32)
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @param ne2 Size of dimension 2
#' @param ne3 Size of dimension 3
#' @return Tensor pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 3, 2)
#' ggml_nelements(t)
#' ggml_free(ctx)
#' }
ggml_new_tensor_4d <- function(ctx, type = GGML_TYPE_F32, ne0, ne1, ne2, ne3) {
  .Call("R_ggml_new_tensor_4d", ctx, as.integer(type),
        as.numeric(ne0), as.numeric(ne1), as.numeric(ne2), as.numeric(ne3))
}

#' Duplicate Tensor
#'
#' Creates a copy of a tensor with the same shape and type
#'
#' @param ctx GGML context
#' @param tensor Tensor to duplicate
#' @return New tensor pointer with same shape
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' b <- ggml_dup_tensor(ctx, a)
#' ggml_nelements(b)
#' ggml_free(ctx)
#' }
ggml_dup_tensor <- function(ctx, tensor) {
  .Call("R_ggml_dup_tensor", ctx, tensor)
}

#' Create Tensor with Arbitrary Dimensions
#'
#' Generic tensor constructor for creating tensors with 1-4 dimensions.
#' This is more flexible than the ggml_new_tensor_Nd functions.
#'
#' @param ctx GGML context
#' @param type Data type (GGML_TYPE_F32, etc.)
#' @param n_dims Number of dimensions (1-4)
#' @param ne Numeric vector of dimension sizes
#' @return Tensor pointer
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor(ctx, GGML_TYPE_F32, 3, c(10, 20, 30))
#' ggml_nelements(t)
#' ggml_free(ctx)
#' }
ggml_new_tensor <- function(ctx, type = GGML_TYPE_F32, n_dims, ne) {
  if (length(ne) < n_dims) {
    stop("ne must have at least n_dims elements")
  }
  .Call("R_ggml_new_tensor", ctx, as.integer(type), as.integer(n_dims), as.numeric(ne))
}

#' Set Tensor to Zero
#'
#' Sets all elements of a tensor to zero.
#' This is more efficient than manually setting all elements.
#'
#' @param tensor Tensor to zero out
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_set_f32(t, 1:10)
#' ggml_set_zero(t)
#' ggml_get_f32(t)
#' ggml_free(ctx)
#' }
ggml_set_zero <- function(tensor) {
  invisible(.Call("R_ggml_set_zero", tensor))
}

#' Set F32 Data
#' @param tensor Tensor
#' @param data Numeric vector
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(t, c(1, 2, 3, 4, 5))
#' ggml_get_f32(t)
#' ggml_free(ctx)
#' }
ggml_set_f32 <- function(tensor, data) {
  invisible(.Call("R_ggml_set_f32", tensor, as.numeric(data)))
}

#' Get F32 Data
#' @param tensor Tensor
#' @return Numeric vector
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(t, c(1, 2, 3, 4, 5))
#' ggml_get_f32(t)
#' ggml_free(ctx)
#' }
ggml_get_f32 <- function(tensor) {
  .Call("R_ggml_get_f32", tensor)
}

#' Set I32 Data
#'
#' Sets integer data in an I32 tensor. Used for indices (ggml_get_rows)
#' and position tensors (ggml_rope).
#'
#' @param tensor Tensor of type GGML_TYPE_I32
#' @param data Integer vector
#' @return NULL (invisible)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10)
#' ggml_set_i32(pos, 0:9)
#' ggml_get_i32(pos)
#' ggml_free(ctx)
#' }
ggml_set_i32 <- function(tensor, data) {
  invisible(.Call("R_ggml_set_i32", tensor, as.integer(data)))
}

#' Get I32 Data
#'
#' Gets integer data from an I32 tensor (e.g., from ggml_argmax)
#'
#' @param tensor Tensor of type GGML_TYPE_I32
#' @return Integer vector
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 10)
#' ggml_set_i32(pos, 0:9)
#' ggml_get_i32(pos)
#' ggml_free(ctx)
#' }
ggml_get_i32 <- function(tensor) {
  .Call("R_ggml_get_i32", tensor)
}

#' Get Number of Elements
#' @param tensor Tensor
#' @return Integer number of elements
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' ggml_nelements(t)
#' ggml_free(ctx)
#' }
ggml_nelements <- function(tensor) {
  .Call("R_ggml_nelements", tensor)
}

#' Get Number of Bytes
#' @param tensor Tensor
#' @return Integer number of bytes
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_nbytes(t)
#' ggml_free(ctx)
#' }
ggml_nbytes <- function(tensor) {
  .Call("R_ggml_nbytes", tensor)
}

# ============================================================================
# Tensor Info Functions
# ============================================================================

#' Get Number of Dimensions
#'
#' Returns the number of dimensions of a tensor
#'
#' @param tensor Tensor pointer
#' @return Integer number of dimensions (1-4)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' ggml_n_dims(t)
#' ggml_free(ctx)
#' }
ggml_n_dims <- function(tensor) {
  .Call("R_ggml_n_dims", tensor)
}

#' Check if Tensor is Contiguous
#'
#' Returns TRUE if tensor data is stored contiguously in memory
#'
#' @param tensor Tensor pointer
#' @return Logical
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_is_contiguous(t)
#' ggml_free(ctx)
#' }
ggml_is_contiguous <- function(tensor) {
  .Call("R_ggml_is_contiguous", tensor)
}

#' Check if Tensor is Transposed
#'
#' Returns TRUE if tensor has been transposed
#'
#' @param tensor Tensor pointer
#' @return Logical
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' ggml_is_transposed(t)
#' ggml_free(ctx)
#' }
ggml_is_transposed <- function(tensor) {
  .Call("R_ggml_is_transposed", tensor)
}

#' Check if Tensor is Permuted
#'
#' Returns TRUE if tensor dimensions have been permuted
#'
#' @param tensor Tensor pointer
#' @return Logical
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' ggml_is_permuted(t)
#' ggml_free(ctx)
#' }
ggml_is_permuted <- function(tensor) {
  .Call("R_ggml_is_permuted", tensor)
}

#' Get Tensor Shape
#'
#' Returns the shape of a tensor as a numeric vector of 4 elements (ne0, ne1, ne2, ne3)
#'
#' @param tensor Tensor pointer
#' @return Numeric vector of length 4 with dimensions
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' ggml_tensor_shape(t)
#' ggml_free(ctx)
#' }
ggml_tensor_shape <- function(tensor) {
  .Call("R_ggml_tensor_shape", tensor)
}

#' Get Tensor Type
#'
#' Returns the data type of a tensor as an integer code
#'
#' @param tensor Tensor pointer
#' @return Integer type code (0 = F32, 1 = F16, etc.)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' t <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' ggml_tensor_type(t)
#' ggml_free(ctx)
#' }
ggml_tensor_type <- function(tensor) {
  .Call("R_ggml_tensor_type", tensor)
}

# ============================================================================
# Scalar Tensor Creation
# ============================================================================

#' Create Scalar I32 Tensor
#'
#' Creates a 1-element tensor containing a single integer value.
#' Useful for indices, counters, and other scalar integer operations.
#'
#' @param ctx GGML context
#' @param value Integer value
#' @return Tensor pointer (1-element I32 tensor)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' scalar <- ggml_new_i32(ctx, 42)
#' ggml_get_i32(scalar)
#' ggml_free(ctx)
#' }
ggml_new_i32 <- function(ctx, value) {
  .Call("R_ggml_new_i32", ctx, as.integer(value), PACKAGE = "ggmlR")
}

#' Create Scalar F32 Tensor
#'
#' Creates a 1-element tensor containing a single float value.
#' Useful for scalar operations, learning rates, and other scalar floats.
#'
#' @param ctx GGML context
#' @param value Numeric value
#' @return Tensor pointer (1-element F32 tensor)
#' @export
#' @examples
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' scalar <- ggml_new_f32(ctx, 3.14)
#' ggml_get_f32(scalar)
#' ggml_free(ctx)
#' }
ggml_new_f32 <- function(ctx, value) {
  .Call("R_ggml_new_f32", ctx, as.numeric(value), PACKAGE = "ggmlR")
}
