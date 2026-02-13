# ============================================================================
# Direct CPU Operations (без графов)
# ============================================================================

#' Element-wise Addition (CPU Direct)
#'
#' Performs element-wise addition of two tensors using direct CPU computation.
#' Returns the result as an R numeric vector. Does NOT use computation graphs.
#'
#' @param a First tensor (must be F32 type)
#' @param b Second tensor (must be F32 type, same size as a)
#' @return Numeric vector containing the element-wise sum
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(5, 4, 3, 2, 1))
#' ggml_cpu_add(a, b)
#' ggml_free(ctx)
#' }
ggml_cpu_add <- function(a, b) {
  .Call("R_ggml_cpu_add", a, b, PACKAGE = "ggmlR")
}

#' Element-wise Multiplication (CPU Direct)
#'
#' Performs element-wise multiplication of two tensors using direct CPU computation.
#' Returns the result as an R numeric vector. Does NOT use computation graphs.
#'
#' @param a First tensor (must be F32 type)
#' @param b Second tensor (must be F32 type, same size as a)
#' @return Numeric vector containing the element-wise product
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(2, 2, 2, 2, 2))
#' ggml_cpu_mul(a, b)
#' ggml_free(ctx)
#' }
ggml_cpu_mul <- function(a, b) {
  .Call("R_ggml_cpu_mul", a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# Graph-based Operations (требуют graph compute)
# ============================================================================

#' Duplicate Tensor (Graph)
#'
#' Creates a graph node that copies a tensor. This is a graph operation
#' that must be computed using ggml_build_forward_expand() and ggml_graph_compute().
#' Unlike ggml_dup_tensor which just allocates, this creates a copy operation in the graph.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the copy operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' b <- ggml_dup(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, b)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(b)
#' ggml_free(ctx)
#' }
ggml_dup <- function(ctx, a) {
  .Call("R_ggml_dup", ctx, a, PACKAGE = "ggmlR")
}

#' Element-wise Addition (Graph)
#'
#' Creates a graph node for element-wise addition. Must be computed using
#' ggml_build_forward_expand() and ggml_graph_compute().
#'
#' @param ctx GGML context
#' @param a First tensor
#' @param b Second tensor (same shape as a)
#' @return Tensor representing the addition operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(5, 4, 3, 2, 1))
#' result <- ggml_add(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_add <- function(ctx, a, b) {
  .Call("R_ggml_add", ctx, a, b, PACKAGE = "ggmlR")
}

#' Add Scalar to Tensor (Graph)
#'
#' Creates a graph node for adding a scalar (1-element tensor) to all elements
#' of a tensor. This is more efficient than creating a full tensor of the same value.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param b Scalar tensor (1-element tensor)
#' @return Tensor representing the operation a + b (broadcasted)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' scalar <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(scalar, 10)
#' result <- ggml_add1(ctx, a, scalar)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_add1 <- function(ctx, a, b) {
  .Call("R_ggml_add1", ctx, a, b, PACKAGE = "ggmlR")
}

#' Element-wise Subtraction (Graph)
#'
#' Creates a graph node for element-wise subtraction.
#'
#' @param ctx GGML context
#' @param a First tensor
#' @param b Second tensor (same shape as a)
#' @return Tensor representing the subtraction operation (a - b)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(5, 4, 3, 2, 1))
#' ggml_set_f32(b, c(1, 1, 1, 1, 1))
#' result <- ggml_sub(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_sub <- function(ctx, a, b) {
  .Call("R_ggml_sub", ctx, a, b, PACKAGE = "ggmlR")
}

#' Element-wise Multiplication (Graph)
#'
#' Creates a graph node for element-wise multiplication.
#'
#' @param ctx GGML context
#' @param a First tensor
#' @param b Second tensor (same shape as a)
#' @return Tensor representing the multiplication operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(2, 2, 2, 2, 2))
#' result <- ggml_mul(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_mul <- function(ctx, a, b) {
  .Call("R_ggml_mul", ctx, a, b, PACKAGE = "ggmlR")
}

#' Element-wise Division (Graph)
#'
#' Creates a graph node for element-wise division.
#'
#' @param ctx GGML context
#' @param a First tensor (numerator)
#' @param b Second tensor (denominator, same shape as a)
#' @return Tensor representing the division operation (a / b)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(10, 20, 30, 40, 50))
#' ggml_set_f32(b, c(2, 2, 2, 2, 2))
#' result <- ggml_div(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_div <- function(ctx, a, b) {
  .Call("R_ggml_div", ctx, a, b, PACKAGE = "ggmlR")
}

#' Matrix Multiplication (Graph)
#'
#' Creates a graph node for matrix multiplication. CRITICAL for LLM operations.
#' For matrices A (m x n) and B (n x p), computes C = A * B (m x p).
#'
#' @param ctx GGML context
#' @param a First matrix tensor
#' @param b Second matrix tensor
#' @return Tensor representing the matrix multiplication
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
#' B <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
#' ggml_set_f32(A, 1:12)
#' ggml_set_f32(B, 1:8)
#' C <- ggml_mul_mat(ctx, A, B)
#' graph <- ggml_build_forward_expand(ctx, C)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(C)
#' ggml_free(ctx)
#' }
ggml_mul_mat <- function(ctx, a, b) {
  .Call("R_ggml_mul_mat", ctx, a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# In-place Operations (Memory-efficient, 2-3x savings for inference)
# ============================================================================

#' Element-wise Addition In-place (Graph)
#'
#' Creates a graph node for in-place element-wise addition.
#' Result is stored in tensor a, saving memory allocation.
#' Returns a view of the modified tensor.
#'
#' @param ctx GGML context
#' @param a First tensor (will be modified in-place)
#' @param b Second tensor (same shape as a)
#' @return View of tensor a with the addition result
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(5, 4, 3, 2, 1))
#' result <- ggml_add_inplace(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_add_inplace <- function(ctx, a, b) {
  .Call("R_ggml_add_inplace", ctx, a, b, PACKAGE = "ggmlR")
}

#' Element-wise Subtraction In-place (Graph)
#'
#' Creates a graph node for in-place element-wise subtraction.
#' Result is stored in tensor a, saving memory allocation.
#'
#' @param ctx GGML context
#' @param a First tensor (will be modified in-place)
#' @param b Second tensor (same shape as a)
#' @return View of tensor a with the subtraction result
#' @export
ggml_sub_inplace <- function(ctx, a, b) {
  .Call("R_ggml_sub_inplace", ctx, a, b, PACKAGE = "ggmlR")
}

#' Element-wise Multiplication In-place (Graph)
#'
#' Creates a graph node for in-place element-wise multiplication.
#' Result is stored in tensor a, saving memory allocation.
#'
#' @param ctx GGML context
#' @param a First tensor (will be modified in-place)
#' @param b Second tensor (same shape as a)
#' @return View of tensor a with the multiplication result
#' @export
ggml_mul_inplace <- function(ctx, a, b) {
  .Call("R_ggml_mul_inplace", ctx, a, b, PACKAGE = "ggmlR")
}

#' Element-wise Division In-place (Graph)
#'
#' Creates a graph node for in-place element-wise division.
#' Result is stored in tensor a, saving memory allocation.
#'
#' @param ctx GGML context
#' @param a First tensor (will be modified in-place)
#' @param b Second tensor (same shape as a)
#' @return View of tensor a with the division result
#' @export
ggml_div_inplace <- function(ctx, a, b) {
  .Call("R_ggml_div_inplace", ctx, a, b, PACKAGE = "ggmlR")
}

#' Square In-place (Graph)
#'
#' Creates a graph node for in-place element-wise square: x^2
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with squared values
#' @export
ggml_sqr_inplace <- function(ctx, a) {
  .Call("R_ggml_sqr_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Square Root In-place (Graph)
#'
#' Creates a graph node for in-place element-wise square root.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with square root values
#' @export
ggml_sqrt_inplace <- function(ctx, a) {
  .Call("R_ggml_sqrt_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Exponential In-place (Graph)
#'
#' Creates a graph node for in-place element-wise exponential: e^x
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with exponential values
#' @export
ggml_exp_inplace <- function(ctx, a) {
  .Call("R_ggml_exp_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Natural Logarithm In-place (Graph)
#'
#' Creates a graph node for in-place element-wise natural logarithm.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with log values
#' @export
ggml_log_inplace <- function(ctx, a) {
  .Call("R_ggml_log_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Absolute Value In-place (Graph)
#'
#' Creates a graph node for in-place element-wise absolute value.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with absolute values
#' @export
ggml_abs_inplace <- function(ctx, a) {
  .Call("R_ggml_abs_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Negation In-place (Graph)
#'
#' Creates a graph node for in-place element-wise negation: -x
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with negated values
#' @export
ggml_neg_inplace <- function(ctx, a) {
  .Call("R_ggml_neg_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Ceiling In-place (Graph)
#'
#' Creates a graph node for in-place element-wise ceiling.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with ceiling values
#' @export
ggml_ceil_inplace <- function(ctx, a) {
  .Call("R_ggml_ceil_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Floor In-place (Graph)
#'
#' Creates a graph node for in-place element-wise floor.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with floor values
#' @export
ggml_floor_inplace <- function(ctx, a) {
  .Call("R_ggml_floor_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Round In-place (Graph)
#'
#' Creates a graph node for in-place element-wise rounding.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with rounded values
#' @export
ggml_round_inplace <- function(ctx, a) {
  .Call("R_ggml_round_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' ReLU Activation In-place (Graph)
#'
#' Creates a graph node for in-place ReLU activation: max(0, x)
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with ReLU applied
#' @export
ggml_relu_inplace <- function(ctx, a) {
  .Call("R_ggml_relu_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' GELU Activation In-place (Graph)
#'
#' Creates a graph node for in-place GELU (Gaussian Error Linear Unit) activation.
#' CRITICAL for GPT models with memory efficiency.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with GELU applied
#' @export
ggml_gelu_inplace <- function(ctx, a) {
  .Call("R_ggml_gelu_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' SiLU Activation In-place (Graph)
#'
#' Creates a graph node for in-place SiLU (Sigmoid Linear Unit) activation.
#' CRITICAL for LLaMA models with memory efficiency.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with SiLU applied
#' @export
ggml_silu_inplace <- function(ctx, a) {
  .Call("R_ggml_silu_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Sigmoid Activation In-place (Graph)
#'
#' Creates a graph node for in-place sigmoid activation: 1 / (1 + e^(-x))
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with sigmoid applied
#' @export
ggml_sigmoid_inplace <- function(ctx, a) {
  .Call("R_ggml_sigmoid_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Tanh Activation In-place (Graph)
#'
#' Creates a graph node for in-place hyperbolic tangent activation.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with tanh applied
#' @export
ggml_tanh_inplace <- function(ctx, a) {
  .Call("R_ggml_tanh_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Softplus Activation In-place (Graph)
#'
#' Creates a graph node for in-place softplus activation: log(1 + e^x)
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with softplus applied
#' @export
ggml_softplus_inplace <- function(ctx, a) {
  .Call("R_ggml_softplus_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' ELU Activation In-place (Graph)
#'
#' Creates a graph node for in-place ELU (Exponential Linear Unit) activation.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of tensor a with ELU applied
#' @export
ggml_elu_inplace <- function(ctx, a) {
  .Call("R_ggml_elu_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Scale Tensor In-place (Graph)
#'
#' Creates a graph node for in-place scaling: a * s
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @param s Scalar value to multiply by
#' @return View of tensor a with scaled values
#' @export
ggml_scale_inplace <- function(ctx, a, s) {
  .Call("R_ggml_scale_inplace", ctx, a, as.numeric(s), PACKAGE = "ggmlR")
}

#' Duplicate Tensor In-place (Graph)
#'
#' Creates a graph node for in-place tensor duplication.
#' Returns a view of the input tensor.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return View of tensor a
#' @export
ggml_dup_inplace <- function(ctx, a) {
  .Call("R_ggml_dup_inplace", ctx, a, PACKAGE = "ggmlR")
}

# ============================================================================
# Activation Functions
# ============================================================================

#' ReLU Activation (Graph)
#'
#' Creates a graph node for ReLU (Rectified Linear Unit) activation: max(0, x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the ReLU operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' result <- ggml_relu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_relu <- function(ctx, a) {
  .Call("R_ggml_relu", ctx, a, PACKAGE = "ggmlR")
}

#' GELU Activation (Graph)
#'
#' Creates a graph node for GELU (Gaussian Error Linear Unit) activation.
#' CRITICAL for GPT models.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the GELU operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' result <- ggml_gelu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_gelu <- function(ctx, a) {
  .Call("R_ggml_gelu", ctx, a, PACKAGE = "ggmlR")
}

#' SiLU Activation (Graph)
#'
#' Creates a graph node for SiLU (Sigmoid Linear Unit) activation, also known as Swish.
#' CRITICAL for LLaMA models.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the SiLU operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' result <- ggml_silu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_silu <- function(ctx, a) {
  .Call("R_ggml_silu", ctx, a, PACKAGE = "ggmlR")
}

#' Tanh Activation (Graph)
#'
#' Creates a graph node for hyperbolic tangent activation.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the tanh operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' result <- ggml_tanh(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_tanh <- function(ctx, a) {
  .Call("R_ggml_tanh", ctx, a, PACKAGE = "ggmlR")
}

# ============================================================================
# Normalization Functions
# ============================================================================

#' Layer Normalization (Graph)
#'
#' Creates a graph node for layer normalization.
#' Normalizes input to zero mean and unit variance.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param eps Epsilon value for numerical stability (default: 1e-5)
#' @return Tensor representing the layer normalization operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_norm(ctx, a, eps = 1e-5)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)  # Normalized values
#' ggml_free(ctx)
#' }
ggml_norm <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_norm", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
}

#' RMS Normalization (Graph)
#'
#' Creates a graph node for RMS (Root Mean Square) normalization.
#' Normalizes by x / sqrt(mean(x^2) + eps). CRITICAL for LLaMA models.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param eps Epsilon value for numerical stability (default: 1e-5)
#' @return Tensor representing the RMS normalization operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_rms_norm(ctx, a, eps = 1e-5)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)
#' # sqrt(mean(output^2)) should be ~1
#' ggml_free(ctx)
#' }
ggml_rms_norm <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_rms_norm", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
}

#' Layer Normalization In-place (Graph)
#'
#' Creates a graph node for in-place layer normalization.
#' Returns a view of the input tensor.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @param eps Epsilon value for numerical stability (default: 1e-5)
#' @return View of input tensor with layer normalization applied
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_norm_inplace(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_norm_inplace <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_norm_inplace", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
}

#' RMS Normalization In-place (Graph)
#'
#' Creates a graph node for in-place RMS normalization.
#' Returns a view of the input tensor.
#' CRITICAL for LLaMA models when memory efficiency is important.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @param eps Epsilon value for numerical stability (default: 1e-5)
#' @return View of input tensor with RMS normalization applied
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_rms_norm_inplace(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_rms_norm_inplace <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_rms_norm_inplace", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
}

#' Group Normalization (Graph)
#'
#' Creates a graph node for group normalization.
#' Normalizes along ne0*ne1*n_groups dimensions.
#' Used in Stable Diffusion and other image generation models.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param n_groups Number of groups to divide channels into
#' @param eps Epsilon for numerical stability (default 1e-5)
#' @return Tensor representing the group norm operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # 4 channels, 2 groups (2 channels per group)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
#' ggml_set_f32(a, rnorm(32))
#' result <- ggml_group_norm(ctx, a, n_groups = 2)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_group_norm <- function(ctx, a, n_groups, eps = 1e-5) {
  .Call("R_ggml_group_norm", ctx, a, as.integer(n_groups), as.numeric(eps), PACKAGE = "ggmlR")
}

#' Group Normalization In-place (Graph)
#'
#' Creates a graph node for in-place group normalization.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @param n_groups Number of groups
#' @param eps Epsilon for numerical stability (default 1e-5)
#' @return View of input tensor with group norm applied
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
#' ggml_set_f32(a, rnorm(32))
#' result <- ggml_group_norm_inplace(ctx, a, n_groups = 2)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_group_norm_inplace <- function(ctx, a, n_groups, eps = 1e-5) {
  .Call("R_ggml_group_norm_inplace", ctx, a, as.integer(n_groups), as.numeric(eps), PACKAGE = "ggmlR")
}

#' L2 Normalization (Graph)
#'
#' Creates a graph node for L2 normalization (unit norm).
#' Normalizes vectors to unit length: x / ||x||_2.
#' Used in RWKV v7 and embedding normalization.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param eps Epsilon for numerical stability (default 1e-5)
#' @return Tensor representing the L2 norm operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(3, 0, 0, 4))  # Length = 5
#' result <- ggml_l2_norm(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [0.6, 0, 0, 0.8] unit vector
#' ggml_free(ctx)
#' }
#' @export
ggml_l2_norm <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_l2_norm", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
}

#' L2 Normalization In-place (Graph)
#'
#' Creates a graph node for in-place L2 normalization.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @param eps Epsilon for numerical stability (default 1e-5)
#' @return View of input tensor with L2 norm applied
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(3, 0, 0, 4))
#' result <- ggml_l2_norm_inplace(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_l2_norm_inplace <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_l2_norm_inplace", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
}

#' RMS Norm Backward (Graph)
#'
#' Creates a graph node for backward pass of RMS normalization.
#' Used in training for computing gradients.
#'
#' @param ctx GGML context
#' @param a Input tensor (x from forward pass)
#' @param b Gradient tensor (dy)
#' @param eps Epsilon for numerical stability (default 1e-5)
#' @return Tensor representing the gradient with respect to input
#' @export
ggml_rms_norm_back <- function(ctx, a, b, eps = 1e-5) {
  .Call("R_ggml_rms_norm_back", ctx, a, b, as.numeric(eps), PACKAGE = "ggmlR")
}

# ============================================================================
# Softmax
# ============================================================================

#' Softmax (Graph)
#'
#' Creates a graph node for softmax operation.
#' CRITICAL for attention mechanisms.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the softmax operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_soft_max(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)
#' # Output sums to 1.0
#' ggml_free(ctx)
#' }
#' @export
ggml_soft_max <- function(ctx, a) {
  .Call("R_ggml_soft_max", ctx, a, PACKAGE = "ggmlR")
}

#' Softmax In-place (Graph)
#'
#' Creates a graph node for in-place softmax operation.
#' Returns a view of the input tensor.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @return View of input tensor with softmax applied
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_soft_max_inplace(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_soft_max_inplace <- function(ctx, a) {
  .Call("R_ggml_soft_max_inplace", ctx, a, PACKAGE = "ggmlR")
}

#' Extended Softmax with Masking and Scaling (Graph)
#'
#' Creates a graph node for fused softmax operation with optional masking
#' and ALiBi (Attention with Linear Biases) support.
#' Computes: softmax(a * scale + mask * (ALiBi slope))
#' CRITICAL for efficient attention computation in transformers.
#'
#' @param ctx GGML context
#' @param a Input tensor (typically attention scores)
#' @param mask Optional attention mask tensor (F16 or F32). NULL for no mask.
#'   Shape must be broadcastable to input tensor.
#' @param scale Scaling factor, typically 1/sqrt(head_dim)
#' @param max_bias Maximum ALiBi bias (0.0 to disable ALiBi)
#' @return Tensor representing the scaled and masked softmax
#'
#' @details
#' This extended softmax is commonly used in transformer attention:
#' 1. Scale attention scores by 1/sqrt(d_k) for numerical stability
#' 2. Apply attention mask (e.g., causal mask, padding mask)
#' 3. Optionally apply ALiBi position bias
#' 4. Compute softmax
#'
#' All these operations are fused for efficiency.
#'
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' scores <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 10)
#' ggml_set_f32(scores, rnorm(100))
#' attn <- ggml_soft_max_ext(ctx, scores, NULL, 1.0, max_bias = 0.0)
#' graph <- ggml_build_forward_expand(ctx, attn)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_soft_max_ext <- function(ctx, a, mask = NULL, scale = 1.0, max_bias = 0.0) {
  .Call("R_ggml_soft_max_ext", ctx, a, mask,
        as.numeric(scale), as.numeric(max_bias), PACKAGE = "ggmlR")
}

#' Extended Softmax Inplace (Graph)
#'
#' Creates a graph node for extended softmax, modifying input tensor in place.
#' Returns a view of the input tensor.
#'
#' @inheritParams ggml_soft_max_ext
#' @return View of input tensor with softmax applied in place
#' @export
#' @family softmax
ggml_soft_max_ext_inplace <- function(ctx, a, mask = NULL, scale = 1.0, max_bias = 0.0) {
  .Call("R_ggml_soft_max_ext_inplace", ctx, a, mask,
        as.numeric(scale), as.numeric(max_bias), PACKAGE = "ggmlR")
}

#' Extended Softmax Backward Inplace (Graph)
#'
#' Creates a graph node for the backward pass of extended softmax, modifying in place.
#'
#' @param ctx GGML context
#' @param a Gradient tensor from upstream
#' @param b Softmax output from forward pass
#' @param scale Scaling factor used in forward pass
#' @param max_bias Maximum ALiBi bias used in forward pass
#' @return View of input tensor with gradient computed in place
#' @export
#' @family softmax
ggml_soft_max_ext_back_inplace <- function(ctx, a, b, scale = 1.0, max_bias = 0.0) {
  .Call("R_ggml_soft_max_ext_back_inplace", ctx, a, b,
        as.numeric(scale), as.numeric(max_bias), PACKAGE = "ggmlR")
}

# ============================================================================
# Basic Operations - Extended
# ============================================================================

#' Transpose (Graph)
#'
#' Creates a graph node for matrix transpose operation.
#'
#' @param ctx GGML context
#' @param a Input tensor (2D matrix)
#' @return Tensor representing the transposed matrix
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
#' ggml_set_f32(a, 1:6)
#' result <- ggml_transpose(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' shape <- ggml_tensor_shape(result)  # [2, 3]
#' ggml_free(ctx)
#' }
#' @export
ggml_transpose <- function(ctx, a) {
  .Call("R_ggml_transpose", ctx, a, PACKAGE = "ggmlR")
}

#' Sum (Graph)
#'
#' Creates a graph node that computes the sum of all elements.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Scalar tensor with the sum
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' result <- ggml_sum(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # 15
#' ggml_free(ctx)
#' }
#' @export
ggml_sum <- function(ctx, a) {
  .Call("R_ggml_sum", ctx, a, PACKAGE = "ggmlR")
}

#' Sum Rows (Graph)
#'
#' Creates a graph node that computes the sum along rows.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor with row sums
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5, 6))
#' result <- ggml_sum_rows(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [6, 15]
#' ggml_free(ctx)
#' }
#' @export
ggml_sum_rows <- function(ctx, a) {
  .Call("R_ggml_sum_rows", ctx, a, PACKAGE = "ggmlR")
}

#' Mean (Graph)
#'
#' Creates a graph node that computes the mean of all elements.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Scalar tensor with the mean
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(2, 4, 6, 8, 10))
#' result <- ggml_mean(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # 6
#' ggml_free(ctx)
#' }
#' @export
ggml_mean <- function(ctx, a) {
  .Call("R_ggml_mean", ctx, a, PACKAGE = "ggmlR")
}

#' Argmax (Graph)
#'
#' Creates a graph node that finds the index of the maximum value.
#' CRITICAL for token generation in LLMs.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor with argmax indices
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 5, 3, 2, 4))
#' result <- ggml_argmax(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_i32(result)  # 1 (0-indexed)
#' ggml_free(ctx)
#' }
#' @export
ggml_argmax <- function(ctx, a) {
  .Call("R_ggml_argmax", ctx, a, PACKAGE = "ggmlR")
}

#' Repeat (Graph)
#'
#' Creates a graph node that repeats tensor 'a' to match shape of tensor 'b'.
#'
#' @param ctx GGML context
#' @param a Tensor to repeat
#' @param b Target tensor (defines output shape)
#' @return Tensor with repeated values
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2)
#' ggml_set_f32(a, c(1, 2))
#' b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2)
#' result <- ggml_repeat(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [1, 1, 1, 2, 2, 2]
#' ggml_free(ctx)
#' }
#' @export
ggml_repeat <- function(ctx, a, b) {
  .Call("R_ggml_repeat", ctx, a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# Additional Activation Functions
# ============================================================================

#' Sigmoid Activation (Graph)
#'
#' Creates a graph node for sigmoid activation: 1 / (1 + exp(-x))
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the sigmoid operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' result <- ggml_sigmoid(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_get_f32(result)
#' ggml_free(ctx)
#' }
ggml_sigmoid <- function(ctx, a) {
  .Call("R_ggml_sigmoid", ctx, a, PACKAGE = "ggmlR")
}

#' GELU Quick Activation (Graph)
#'
#' Creates a graph node for fast approximation of GELU.
#' Faster than standard GELU with minimal accuracy loss.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the GELU quick operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' result <- ggml_gelu_quick(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)
#' ggml_free(ctx)
#' }
#' @export
ggml_gelu_quick <- function(ctx, a) {
  .Call("R_ggml_gelu_quick", ctx, a, PACKAGE = "ggmlR")
}

#' ELU Activation (Graph)
#'
#' Creates a graph node for ELU (Exponential Linear Unit) activation.
#' ELU(x) = x if x > 0, else alpha * (exp(x) - 1) where alpha = 1.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the ELU operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' r <- ggml_elu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)
#' ggml_free(ctx)
#' }
ggml_elu <- function(ctx, a) {
  .Call("R_ggml_elu", ctx, a, PACKAGE = "ggmlR")
}

#' Leaky ReLU Activation (Graph)
#'
#' Creates a graph node for Leaky ReLU activation.
#' LeakyReLU(x) = x if x > 0, else negative_slope * x.
#' Unlike standard ReLU, Leaky ReLU allows a small gradient for negative values.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param negative_slope Slope for negative values (default: 0.01)
#' @param inplace If TRUE, operation is performed in-place (default: FALSE)
#' @return Tensor representing the Leaky ReLU operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' r <- ggml_leaky_relu(ctx, a, negative_slope = 0.1)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # [-0.2, -0.1, 0, 1, 2]
#' ggml_free(ctx)
#' }
ggml_leaky_relu <- function(ctx, a, negative_slope = 0.01, inplace = FALSE) {
  .Call("R_ggml_leaky_relu", ctx, a, as.numeric(negative_slope),
        as.logical(inplace), PACKAGE = "ggmlR")
}

#' Hard Swish Activation (Graph)
#'
#' Creates a graph node for Hard Swish activation.
#' HardSwish(x) = x * ReLU6(x + 3) / 6 = x * min(max(0, x + 3), 6) / 6.
#' Used in MobileNetV3 and other efficient architectures.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the Hard Swish operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-4, -1, 0, 1, 4))
#' r <- ggml_hardswish(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)
#' ggml_free(ctx)
#' }
ggml_hardswish <- function(ctx, a) {
  .Call("R_ggml_hardswish", ctx, a, PACKAGE = "ggmlR")
}

#' Hard Sigmoid Activation (Graph)
#'
#' Creates a graph node for Hard Sigmoid activation.
#' HardSigmoid(x) = ReLU6(x + 3) / 6 = min(max(0, x + 3), 6) / 6.
#' A computationally efficient approximation of the sigmoid function.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the Hard Sigmoid operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-4, -1, 0, 1, 4))
#' r <- ggml_hardsigmoid(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # [0, 0.333, 0.5, 0.667, 1]
#' ggml_free(ctx)
#' }
ggml_hardsigmoid <- function(ctx, a) {
  .Call("R_ggml_hardsigmoid", ctx, a, PACKAGE = "ggmlR")
}

#' Softplus Activation (Graph)
#'
#' Creates a graph node for Softplus activation.
#' Softplus(x) = log(1 + exp(x)).
#' A smooth approximation of ReLU.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the Softplus operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' r <- ggml_softplus(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)
#' ggml_free(ctx)
#' }
ggml_softplus <- function(ctx, a) {
  .Call("R_ggml_softplus", ctx, a, PACKAGE = "ggmlR")
}

#' Exact GELU Activation (Graph)
#'
#' Creates a graph node for exact GELU using the error function (erf).
#' GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))).
#' More accurate than approximate GELU but potentially slower on some backends.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the exact GELU operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -1, 0, 1, 2))
#' r <- ggml_gelu_erf(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)
#' ggml_free(ctx)
#' }
ggml_gelu_erf <- function(ctx, a) {
  .Call("R_ggml_gelu_erf", ctx, a, PACKAGE = "ggmlR")
}

# ============================================================================
# View/Reshape Operations
# ============================================================================

#' View Tensor
#'
#' Creates a view of the tensor (shares data, no copy)
#'
#' @param ctx GGML context
#' @param src Source tensor
#' @return View tensor (shares data with src)
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10)
#' view <- ggml_view_tensor(ctx, a)
#' # view shares data with a
#' ggml_free(ctx)
#' }
#' @export
ggml_view_tensor <- function(ctx, src) {
  .Call("R_ggml_view_tensor", ctx, src, PACKAGE = "ggmlR")
}

#' Reshape to 1D (Graph)
#'
#' Reshapes tensor to 1D with ne0 elements
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param ne0 Size of dimension 0
#' @return Reshaped tensor
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
#' ggml_set_f32(a, 1:12)
#' result <- ggml_reshape_1d(ctx, a, 12)
#' ggml_free(ctx)
#' }
#' @export
ggml_reshape_1d <- function(ctx, a, ne0) {
  .Call("R_ggml_reshape_1d", ctx, a, as.numeric(ne0), PACKAGE = "ggmlR")
}

#' Reshape to 2D (Graph)
#'
#' Reshapes tensor to 2D with shape (ne0, ne1)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @return Reshaped tensor
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 12)
#' ggml_set_f32(a, 1:12)
#' result <- ggml_reshape_2d(ctx, a, 3, 4)
#' ggml_free(ctx)
#' }
#' @export
ggml_reshape_2d <- function(ctx, a, ne0, ne1) {
  .Call("R_ggml_reshape_2d", ctx, a, as.numeric(ne0), as.numeric(ne1), PACKAGE = "ggmlR")
}

#' Reshape to 3D (Graph)
#'
#' Reshapes tensor to 3D with shape (ne0, ne1, ne2)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @param ne2 Size of dimension 2
#' @return Reshaped tensor
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 24)
#' ggml_set_f32(a, 1:24)
#' result <- ggml_reshape_3d(ctx, a, 2, 3, 4)
#' ggml_free(ctx)
#' }
#' @export
ggml_reshape_3d <- function(ctx, a, ne0, ne1, ne2) {
  .Call("R_ggml_reshape_3d", ctx, a, as.numeric(ne0), as.numeric(ne1),
        as.numeric(ne2), PACKAGE = "ggmlR")
}

#' Reshape to 4D (Graph)
#'
#' Reshapes tensor to 4D with shape (ne0, ne1, ne2, ne3)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @param ne2 Size of dimension 2
#' @param ne3 Size of dimension 3
#' @return Reshaped tensor
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 120)
#' ggml_set_f32(a, 1:120)
#' result <- ggml_reshape_4d(ctx, a, 2, 3, 4, 5)
#' ggml_free(ctx)
#' }
#' @export
ggml_reshape_4d <- function(ctx, a, ne0, ne1, ne2, ne3) {
  .Call("R_ggml_reshape_4d", ctx, a, as.numeric(ne0), as.numeric(ne1),
        as.numeric(ne2), as.numeric(ne3), PACKAGE = "ggmlR")
}

#' Permute Tensor Dimensions (Graph)
#'
#' Permutes the tensor dimensions according to specified axes.
#' CRITICAL for attention mechanisms in transformers.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param axis0 New position for axis 0
#' @param axis1 New position for axis 1
#' @param axis2 New position for axis 2
#' @param axis3 New position for axis 3
#' @return Permuted tensor
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Create 4D tensor: (2, 3, 4, 5)
#' t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 3, 4, 5)
#' # Swap axes 0 and 1: result shape (3, 2, 4, 5)
#' t_perm <- ggml_permute(ctx, t, 1, 0, 2, 3)
#' ggml_free(ctx)
#' }
ggml_permute <- function(ctx, a, axis0, axis1, axis2, axis3) {
  .Call("R_ggml_permute", ctx, a, as.integer(axis0), as.integer(axis1),
        as.integer(axis2), as.integer(axis3), PACKAGE = "ggmlR")
}

#' Make Contiguous (Graph)
#'
#' Makes a tensor contiguous in memory. Required after permute/transpose
#' before some operations.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Contiguous tensor
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4)
#' ggml_set_f32(a, 1:12)
#' transposed <- ggml_transpose(ctx, a)
#' contiguous <- ggml_cont(ctx, transposed)
#' ggml_free(ctx)
#' }
#' @export
ggml_cont <- function(ctx, a) {
  .Call("R_ggml_cont", ctx, a, PACKAGE = "ggmlR")
}

# ============================================================================
# Mathematical Operations
# ============================================================================

#' Square (Graph)
#'
#' Creates a graph node for element-wise squaring: x^2
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the square operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_sqr(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [1, 4, 9, 16]
#' ggml_free(ctx)
#' }
#' @export
ggml_sqr <- function(ctx, a) {
  .Call("R_ggml_sqr", ctx, a, PACKAGE = "ggmlR")
}

#' Square Root (Graph)
#'
#' Creates a graph node for element-wise square root: sqrt(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the sqrt operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 4, 9, 16))
#' result <- ggml_sqrt(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [1, 2, 3, 4]
#' ggml_free(ctx)
#' }
#' @export
ggml_sqrt <- function(ctx, a) {
  .Call("R_ggml_sqrt", ctx, a, PACKAGE = "ggmlR")
}

#' Natural Logarithm (Graph)
#'
#' Creates a graph node for element-wise natural logarithm: log(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the log operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
#' ggml_set_f32(a, c(1, exp(1), exp(2)))
#' result <- ggml_log(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [0, 1, 2]
#' ggml_free(ctx)
#' }
#' @export
ggml_log <- function(ctx, a) {
  .Call("R_ggml_log", ctx, a, PACKAGE = "ggmlR")
}

#' Exponential (Graph)
#'
#' Creates a graph node for element-wise exponential: exp(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the exp operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
#' ggml_set_f32(a, c(0, 1, 2))
#' result <- ggml_exp(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [1, e, e^2]
#' ggml_free(ctx)
#' }
#' @export
ggml_exp <- function(ctx, a) {
  .Call("R_ggml_exp", ctx, a, PACKAGE = "ggmlR")
}

#' Absolute Value (Graph)
#'
#' Creates a graph node for element-wise absolute value: |x|
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the abs operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(-2, -1, 1, 2))
#' result <- ggml_abs(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [2, 1, 1, 2]
#' ggml_free(ctx)
#' }
#' @export
ggml_abs <- function(ctx, a) {
  .Call("R_ggml_abs", ctx, a, PACKAGE = "ggmlR")
}

#' Negation (Graph)
#'
#' Creates a graph node for element-wise negation: -x
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the negation operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, -2, 3, -4))
#' result <- ggml_neg(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [-1, 2, -3, 4]
#' ggml_free(ctx)
#' }
#' @export
ggml_neg <- function(ctx, a) {
  .Call("R_ggml_neg", ctx, a, PACKAGE = "ggmlR")
}

#' Sign Function (Graph)
#'
#' Creates a graph node for element-wise sign function.
#' sgn(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the sign operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -0.5, 0, 0.5, 2))
#' r <- ggml_sgn(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # c(-1, -1, 0, 1, 1)
#' ggml_free(ctx)
#' }
ggml_sgn <- function(ctx, a) {
  .Call("R_ggml_sgn", ctx, a, PACKAGE = "ggmlR")
}

#' Step Function (Graph)
#'
#' Creates a graph node for element-wise step function.
#' step(x) = 0 if x <= 0, 1 if x > 0
#' Also known as the Heaviside step function.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the step operation
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(-2, -0.5, 0, 0.5, 2))
#' r <- ggml_step(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # c(0, 0, 0, 1, 1)
#' ggml_free(ctx)
#' }
ggml_step <- function(ctx, a) {
  .Call("R_ggml_step", ctx, a, PACKAGE = "ggmlR")
}

#' Sine (Graph)
#'
#' Creates a graph node for element-wise sine: sin(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the sin operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(0, pi/6, pi/2, pi))
#' result <- ggml_sin(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [0, 0.5, 1, 0]
#' ggml_free(ctx)
#' }
#' @export
ggml_sin <- function(ctx, a) {
  .Call("R_ggml_sin", ctx, a, PACKAGE = "ggmlR")
}

#' Cosine (Graph)
#'
#' Creates a graph node for element-wise cosine: cos(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the cos operation
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(0, pi/3, pi/2, pi))
#' result <- ggml_cos(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [1, 0.5, 0, -1]
#' ggml_free(ctx)
#' }
#' @export
ggml_cos <- function(ctx, a) {
  .Call("R_ggml_cos", ctx, a, PACKAGE = "ggmlR")
}

#' Scale (Graph)
#'
#' Creates a graph node for scaling tensor by a scalar: x * s
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param s Scalar value to multiply by
#' @return Tensor representing the scaled values
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3, 4))
#' result <- ggml_scale(ctx, a, 2.0)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' output <- ggml_get_f32(result)  # [2, 4, 6, 8]
#' ggml_free(ctx)
#' }
#' @export
ggml_scale <- function(ctx, a, s) {
  .Call("R_ggml_scale", ctx, a, as.numeric(s), PACKAGE = "ggmlR")
}

#' Clamp (Graph)
#'
#' Creates a graph node for clamping values to a range: clamp(x, min, max)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param min_val Minimum value
#' @param max_val Maximum value
#' @return Tensor with values clamped to [min_val, max_val]
#' @export
ggml_clamp <- function(ctx, a, min_val, max_val) {
  .Call("R_ggml_clamp", ctx, a, as.numeric(min_val), as.numeric(max_val), PACKAGE = "ggmlR")
}

#' Floor (Graph)
#'
#' Creates a graph node for element-wise floor: floor(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the floor operation
#' @export
ggml_floor <- function(ctx, a) {
  .Call("R_ggml_floor", ctx, a, PACKAGE = "ggmlR")
}

#' Ceiling (Graph)
#'
#' Creates a graph node for element-wise ceiling: ceil(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the ceil operation
#' @export
ggml_ceil <- function(ctx, a) {
  .Call("R_ggml_ceil", ctx, a, PACKAGE = "ggmlR")
}

#' Round (Graph)
#'
#' Creates a graph node for element-wise rounding: round(x)
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @return Tensor representing the round operation
#' @export
ggml_round <- function(ctx, a) {
  .Call("R_ggml_round", ctx, a, PACKAGE = "ggmlR")
}

# ============================================================================
# GLU (Gated Linear Unit) Operations
# ============================================================================

#' GLU Operation Types
#'
#' Constants for GLU (Gated Linear Unit) operation types.
#' Used with ggml_glu() and ggml_glu_split().
#'
#' @format Integer constants
#' @return An integer constant representing a GLU operation type
#' @details
#' \itemize{
#'   \item \code{GGML_GLU_OP_REGLU} (0): ReGLU - ReLU gating
#'   \item \code{GGML_GLU_OP_GEGLU} (1): GeGLU - GELU gating (used in GPT-NeoX, Falcon)
#'   \item \code{GGML_GLU_OP_SWIGLU} (2): SwiGLU - SiLU/Swish gating (used in LLaMA, Mistral)
#'   \item \code{GGML_GLU_OP_SWIGLU_OAI} (3): SwiGLU OpenAI variant
#'   \item \code{GGML_GLU_OP_GEGLU_ERF} (4): GeGLU with exact erf implementation
#'   \item \code{GGML_GLU_OP_GEGLU_QUICK} (5): GeGLU with fast approximation
#' }
#' @examples
#' \donttest{
#' GGML_GLU_OP_REGLU       # 0 - ReLU gating
#' GGML_GLU_OP_GEGLU       # 1 - GELU gating
#' GGML_GLU_OP_SWIGLU      # 2 - SiLU/Swish gating
#' GGML_GLU_OP_SWIGLU_OAI  # 3 - SwiGLU OpenAI
#' GGML_GLU_OP_GEGLU_ERF   # 4 - GELU with erf
#' GGML_GLU_OP_GEGLU_QUICK # 5 - Fast GELU
#' }
#' @export
GGML_GLU_OP_REGLU <- 0L

#' @rdname GGML_GLU_OP_REGLU
#' @export
GGML_GLU_OP_GEGLU <- 1L

#' @rdname GGML_GLU_OP_REGLU
#' @export
GGML_GLU_OP_SWIGLU <- 2L

#' @rdname GGML_GLU_OP_REGLU
#' @export
GGML_GLU_OP_SWIGLU_OAI <- 3L

#' @rdname GGML_GLU_OP_REGLU
#' @export
GGML_GLU_OP_GEGLU_ERF <- 4L

#' @rdname GGML_GLU_OP_REGLU
#' @export
GGML_GLU_OP_GEGLU_QUICK <- 5L

#' Generic GLU (Gated Linear Unit) (Graph)
#'
#' Creates a graph node for GLU operation with specified gating type.
#' GLU splits the input tensor in half along the first dimension,
#' applies an activation to the first half (x), and multiplies it with the second half (gate).
#'
#' Formula: output = activation(x) * gate
#' where x and gate are the two halves of the input tensor.
#'
#' @param ctx GGML context
#' @param a Input tensor (first dimension must be even)
#' @param op GLU operation type (GGML_GLU_OP_REGLU, GGML_GLU_OP_GEGLU, etc.)
#' @param swapped If TRUE, swap x and gate halves (default FALSE)
#' @return Tensor with shape [n/2, ...] where n is the first dimension of input
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Create tensor with 10 columns (will be split into 5 + 5)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 4)
#' ggml_set_f32(a, rnorm(40))
#' # Apply SwiGLU
#' r <- ggml_glu(ctx, a, GGML_GLU_OP_SWIGLU, FALSE)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # Shape: 5x4
#' ggml_free(ctx)
#' }
ggml_glu <- function(ctx, a, op, swapped = FALSE) {
  .Call("R_ggml_glu", ctx, a, as.integer(op), as.logical(swapped), PACKAGE = "ggmlR")
}

#' ReGLU (ReLU Gated Linear Unit) (Graph)
#'
#' Creates a graph node for ReGLU operation.
#' ReGLU uses ReLU as the activation function on the first half.
#'
#' Formula: output = ReLU(x) * gate
#'
#' @param ctx GGML context
#' @param a Input tensor (first dimension must be even)
#' @return Tensor with half the first dimension of input
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3)
#' ggml_set_f32(a, rnorm(24))
#' r <- ggml_reglu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # Shape: 4x3
#' ggml_free(ctx)
#' }
ggml_reglu <- function(ctx, a) {
  .Call("R_ggml_reglu", ctx, a, PACKAGE = "ggmlR")
}

#' GeGLU (GELU Gated Linear Unit) (Graph)
#'
#' Creates a graph node for GeGLU operation.
#' GeGLU uses GELU as the activation function on the first half.
#' CRITICAL for models like GPT-NeoX and Falcon.
#'
#' Formula: output = GELU(x) * gate
#'
#' @param ctx GGML context
#' @param a Input tensor (first dimension must be even)
#' @return Tensor with half the first dimension of input
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3)
#' ggml_set_f32(a, rnorm(24))
#' r <- ggml_geglu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # Shape: 4x3
#' ggml_free(ctx)
#' }
ggml_geglu <- function(ctx, a) {
  .Call("R_ggml_geglu", ctx, a, PACKAGE = "ggmlR")
}

#' SwiGLU (Swish/SiLU Gated Linear Unit) (Graph)
#'
#' Creates a graph node for SwiGLU operation.
#' SwiGLU uses SiLU (Swish) as the activation function on the first half.
#' CRITICAL for LLaMA, Mistral, and many modern LLMs.
#'
#' Formula: output = SiLU(x) * gate
#'
#' @param ctx GGML context
#' @param a Input tensor (first dimension must be even)
#' @return Tensor with half the first dimension of input
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3)
#' ggml_set_f32(a, rnorm(24))
#' r <- ggml_swiglu(ctx, a)
#' graph <- ggml_build_forward_expand(ctx, r)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(r)  # Shape: 4x3
#' ggml_free(ctx)
#' }
ggml_swiglu <- function(ctx, a) {
  .Call("R_ggml_swiglu", ctx, a, PACKAGE = "ggmlR")
}

#' GeGLU Quick (Fast GeGLU) (Graph)
#'
#' Creates a graph node for fast GeGLU approximation.
#' Uses faster but less accurate GELU approximation for gating.
#'
#' @param ctx GGML context
#' @param a Input tensor (first dimension must be even)
#' @return Tensor with half the first dimension of input
#' @export
ggml_geglu_quick <- function(ctx, a) {
  .Call("R_ggml_geglu_quick", ctx, a, PACKAGE = "ggmlR")
}

#' Generic GLU Split (Graph)
#'
#' Creates a graph node for GLU with separate input and gate tensors.
#' Unlike standard GLU which splits a single tensor, this takes two separate tensors.
#'
#' @param ctx GGML context
#' @param a Input tensor (the values to be gated)
#' @param b Gate tensor (same shape as a)
#' @param op GLU operation type (GGML_GLU_OP_REGLU, GGML_GLU_OP_GEGLU, etc.)
#' @return Tensor with same shape as input tensors
#' @export
ggml_glu_split <- function(ctx, a, b, op) {
  .Call("R_ggml_glu_split", ctx, a, b, as.integer(op), PACKAGE = "ggmlR")
}

#' ReGLU Split (Graph)
#'
#' Creates a graph node for ReGLU with separate input and gate tensors.
#'
#' Formula: output = ReLU(a) * b
#'
#' @param ctx GGML context
#' @param a Input tensor (the values to be gated)
#' @param b Gate tensor (same shape as a)
#' @return Tensor with same shape as input tensors
#' @export
ggml_reglu_split <- function(ctx, a, b) {
  .Call("R_ggml_reglu_split", ctx, a, b, PACKAGE = "ggmlR")
}

#' GeGLU Split (Graph)
#'
#' Creates a graph node for GeGLU with separate input and gate tensors.
#'
#' Formula: output = GELU(a) * b
#'
#' @param ctx GGML context
#' @param a Input tensor (the values to be gated)
#' @param b Gate tensor (same shape as a)
#' @return Tensor with same shape as input tensors
#' @export
ggml_geglu_split <- function(ctx, a, b) {
  .Call("R_ggml_geglu_split", ctx, a, b, PACKAGE = "ggmlR")
}

#' SwiGLU Split (Graph)
#'
#' Creates a graph node for SwiGLU with separate input and gate tensors.
#'
#' Formula: output = SiLU(a) * b
#'
#' @param ctx GGML context
#' @param a Input tensor (the values to be gated)
#' @param b Gate tensor (same shape as a)
#' @return Tensor with same shape as input tensors
#' @export
ggml_swiglu_split <- function(ctx, a, b) {
  .Call("R_ggml_swiglu_split", ctx, a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# Row Operations
# ============================================================================

#' Get Rows by Indices (Graph)
#'
#' Creates a graph node that extracts rows from a tensor by index.
#' This is commonly used for embedding lookup in LLMs.
#'
#' @param ctx GGML context
#' @param a Data tensor of shape [n_embd, n_rows, ...] - the embedding table
#' @param b Index tensor (int32) of shape [n_indices] - which rows to extract
#' @return Tensor of shape [n_embd, n_indices, ...] containing the selected rows
#'
#' @details
#' This operation is fundamental for embedding lookup in transformers:
#' given a vocabulary embedding matrix and token indices, it retrieves
#' the corresponding embedding vectors.
#'
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Create embedding matrix: 10 tokens, 4-dim embeddings
#' embeddings <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10)
#' ggml_set_f32(embeddings, rnorm(40))
#' # Token indices to look up
#' indices <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)
#' ggml_set_i32(indices, c(0L, 5L, 2L))
#' # Get embeddings for tokens 0, 5, 2
#' result <- ggml_get_rows(ctx, embeddings, indices)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_get_rows <- function(ctx, a, b) {
  .Call("R_ggml_get_rows", ctx, a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# Diagonal Masking Operations (for Causal Attention)
# ============================================================================

#' Diagonal Mask with -Inf (Graph)
#'
#' Creates a graph node that sets elements above the diagonal to -Inf.
#' This is used for causal (autoregressive) attention masking.
#'
#' @param ctx GGML context
#' @param a Input tensor (typically attention scores)
#' @param n_past Number of past tokens (shifts the diagonal). Use 0 for
#'   standard causal masking where position i can only attend to positions <= i.
#' @return Tensor with same shape as input, elements above diagonal set to -Inf
#'
#' @details
#' In causal attention, we want each position to only attend to itself and
#' previous positions. Setting future positions to -Inf ensures that after
#' softmax, they contribute 0 attention weight.
#'
#' The n_past parameter allows for KV-cache scenarios where the diagonal
#' needs to be shifted to account for previously processed tokens.
#'
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Create attention scores matrix
#' scores <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
#' ggml_set_f32(scores, rep(1, 16))
#' # Apply causal mask
#' masked <- ggml_diag_mask_inf(ctx, scores, 0)
#' graph <- ggml_build_forward_expand(ctx, masked)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_diag_mask_inf <- function(ctx, a, n_past) {
  .Call("R_ggml_diag_mask_inf", ctx, a, as.integer(n_past), PACKAGE = "ggmlR")
}

#' Diagonal Mask with -Inf In-place (Graph)
#'
#' In-place version of ggml_diag_mask_inf. Returns a view of the input tensor.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @param n_past Number of past tokens
#' @return View of input tensor with elements above diagonal set to -Inf
#' @export
ggml_diag_mask_inf_inplace <- function(ctx, a, n_past) {
  .Call("R_ggml_diag_mask_inf_inplace", ctx, a, as.integer(n_past), PACKAGE = "ggmlR")
}

#' Diagonal Mask with Zero (Graph)
#'
#' Creates a graph node that sets elements above the diagonal to 0.
#' Alternative to -Inf masking for certain use cases.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param n_past Number of past tokens
#' @return Tensor with same shape as input, elements above diagonal set to 0
#' @export
ggml_diag_mask_zero <- function(ctx, a, n_past) {
  .Call("R_ggml_diag_mask_zero", ctx, a, as.integer(n_past), PACKAGE = "ggmlR")
}

# ============================================================================
# RoPE - Rotary Position Embedding
# ============================================================================

#' RoPE Mode Constants
#'
#' RoPE (Rotary Position Embedding) Type Constants
#'
#' Constants for RoPE (Rotary Position Embedding) modes used in transformer models.
#' Different models use different RoPE implementations.
#'
#' @format Integer constants
#' @return An integer constant representing a RoPE type
#' @details
#' \itemize{
#'   \item \code{GGML_ROPE_TYPE_NORM} (0): Standard RoPE as in original paper (LLaMA, Mistral)
#'   \item \code{GGML_ROPE_TYPE_NEOX} (2): GPT-NeoX style RoPE with different interleaving
#'   \item \code{GGML_ROPE_TYPE_MROPE} (8): Multi-RoPE for multimodal models (Qwen2-VL)
#'   \item \code{GGML_ROPE_TYPE_VISION} (24): Vision model RoPE variant
#' }
#' @examples
#' \donttest{
#' GGML_ROPE_TYPE_NORM    # 0 - Standard RoPE (LLaMA, Mistral)
#' GGML_ROPE_TYPE_NEOX    # 2 - GPT-NeoX style
#' GGML_ROPE_TYPE_MROPE   # 8 - Multi-RoPE (Qwen2-VL)
#' GGML_ROPE_TYPE_VISION  # 24 - Vision models
#' }
#' @name rope_types
#' @rdname rope_types
#' @export
GGML_ROPE_TYPE_NORM <- 0L

#' @rdname rope_types
#' @export
GGML_ROPE_TYPE_NEOX <- 2L

#' @rdname rope_types
#' @export
GGML_ROPE_TYPE_MROPE <- 8L

#' @rdname rope_types
#' @export
GGML_ROPE_TYPE_VISION <- 24L

#' Rotary Position Embedding (Graph)
#'
#' Creates a graph node for RoPE (Rotary Position Embedding).
#' RoPE is the dominant position encoding method in modern LLMs like LLaMA,
#' Mistral, and many others.
#'
#' @param ctx GGML context
#' @param a Input tensor of shape [head_dim, n_head, seq_len, batch]
#' @param b Position tensor (int32) of shape [seq_len] containing position indices
#' @param n_dims Number of dimensions to apply rotation to (usually head_dim)
#' @param mode RoPE mode: GGML_ROPE_TYPE_NORM (0), GGML_ROPE_TYPE_NEOX (2), etc.
#' @return Tensor with same shape as input, with rotary embeddings applied
#'
#' @details
#' RoPE encodes position information by rotating pairs of dimensions in the
#' embedding space. The rotation angle depends on position and dimension index.
#'
#' Key benefits of RoPE:
#' - Relative position information emerges naturally from rotation
#' - Better extrapolation to longer sequences than absolute embeddings
#' - No additional parameters needed
#'
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Query tensor: head_dim=8, n_head=4, seq_len=16, batch=1
#' q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 4, 16, 1)
#' ggml_set_f32(q, rnorm(8 * 4 * 16))
#' # Position indices
#' pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16)
#' ggml_set_i32(pos, 0:15)
#' # Apply RoPE
#' q_rope <- ggml_rope(ctx, q, pos, 8, GGML_ROPE_TYPE_NORM)
#' graph <- ggml_build_forward_expand(ctx, q_rope)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_rope <- function(ctx, a, b, n_dims, mode = 0L) {
  .Call("R_ggml_rope", ctx, a, b, as.integer(n_dims), as.integer(mode), PACKAGE = "ggmlR")
}

#' Rotary Position Embedding In-place (Graph)
#'
#' In-place version of ggml_rope. Returns a view of the input tensor.
#'
#' @param ctx GGML context
#' @param a Input tensor (will be modified in-place)
#' @param b Position tensor (int32)
#' @param n_dims Number of dimensions to apply rotation to
#' @param mode RoPE mode
#' @return View of input tensor with RoPE applied
#' @export
ggml_rope_inplace <- function(ctx, a, b, n_dims, mode = 0L) {
  .Call("R_ggml_rope_inplace", ctx, a, b, as.integer(n_dims), as.integer(mode), PACKAGE = "ggmlR")
}

#' Extended RoPE with Frequency Scaling (Graph)
#'
#' Creates a graph node for extended RoPE with frequency scaling parameters.
#' Supports context extension techniques like YaRN, Linear Scaling, etc.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param b Position tensor (int32)
#' @param c Optional frequency factors tensor (NULL for default)
#' @param n_dims Number of dimensions to apply rotation to
#' @param mode RoPE mode
#' @param n_ctx_orig Original context length the model was trained on
#' @param freq_base Base frequency for RoPE (default 10000 for most models)
#' @param freq_scale Frequency scale factor (1.0 = no scaling)
#' @param ext_factor YaRN extension factor (0.0 to disable)
#' @param attn_factor Attention scale factor (typically 1.0)
#' @param beta_fast YaRN parameter for fast dimensions
#' @param beta_slow YaRN parameter for slow dimensions
#' @return Tensor with extended RoPE applied
#'
#' @details
#' This extended version supports various context extension techniques:
#'
#' - **Linear Scaling**: Set freq_scale = original_ctx / new_ctx
#' - **YaRN**: Set ext_factor > 0 with appropriate beta_fast/beta_slow
#' - **NTK-aware**: Adjust freq_base for NTK-style scaling
#'
#' Common freq_base values:
#' - LLaMA 1/2: 10000
#' - LLaMA 3: 500000
#' - Mistral: 10000
#' - Phi-3: 10000
#'
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 32, 1)
#' ggml_set_f32(q, rnorm(64 * 8 * 32))
#' pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 32)
#' ggml_set_i32(pos, 0:31)
#' # Standard RoPE with default freq_base
#' q_rope <- ggml_rope_ext(ctx, q, pos, NULL,
#'                         n_dims = 64, mode = 0L,
#'                         n_ctx_orig = 4096,
#'                         freq_base = 10000, freq_scale = 1.0,
#'                         ext_factor = 0.0, attn_factor = 1.0,
#'                         beta_fast = 32, beta_slow = 1)
#' graph <- ggml_build_forward_expand(ctx, q_rope)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_rope_ext <- function(ctx, a, b, c = NULL,
                          n_dims, mode = 0L, n_ctx_orig = 0L,
                          freq_base = 10000.0, freq_scale = 1.0,
                          ext_factor = 0.0, attn_factor = 1.0,
                          beta_fast = 32.0, beta_slow = 1.0) {
  .Call("R_ggml_rope_ext", ctx, a, b, c,
        as.integer(n_dims), as.integer(mode), as.integer(n_ctx_orig),
        as.numeric(freq_base), as.numeric(freq_scale),
        as.numeric(ext_factor), as.numeric(attn_factor),
        as.numeric(beta_fast), as.numeric(beta_slow),
        PACKAGE = "ggmlR")
}

#' Extended RoPE Inplace (Graph)
#'
#' Creates a graph node for extended RoPE, modifying input tensor in place.
#' Returns a view of the input tensor.
#'
#' @inheritParams ggml_rope_ext
#' @return View of input tensor with RoPE applied in place
#' @export
#' @family rope
ggml_rope_ext_inplace <- function(ctx, a, b, c = NULL,
                                   n_dims, mode = 0L, n_ctx_orig = 0L,
                                   freq_base = 10000.0, freq_scale = 1.0,
                                   ext_factor = 0.0, attn_factor = 1.0,
                                   beta_fast = 32.0, beta_slow = 1.0) {
  .Call("R_ggml_rope_ext_inplace", ctx, a, b, c,
        as.integer(n_dims), as.integer(mode), as.integer(n_ctx_orig),
        as.numeric(freq_base), as.numeric(freq_scale),
        as.numeric(ext_factor), as.numeric(attn_factor),
        as.numeric(beta_fast), as.numeric(beta_slow),
        PACKAGE = "ggmlR")
}

#' Multi-RoPE for Vision Models (Graph)
#'
#' Creates a graph node for multi-dimensional RoPE (MRoPE) used in vision transformers.
#' Supports separate rotation for different positional dimensions (e.g., height, width, time).
#'
#' @inheritParams ggml_rope_ext
#' @param sections Integer vector of length 4 specifying dimension sections for MRoPE
#' @return Tensor with multi-dimensional RoPE applied
#' @export
#' @family rope
ggml_rope_multi <- function(ctx, a, b, c = NULL,
                             n_dims, sections = c(0L, 0L, 0L, 0L),
                             mode = 0L, n_ctx_orig = 0L,
                             freq_base = 10000.0, freq_scale = 1.0,
                             ext_factor = 0.0, attn_factor = 1.0,
                             beta_fast = 32.0, beta_slow = 1.0) {
  .Call("R_ggml_rope_multi", ctx, a, b, c,
        as.integer(n_dims), as.integer(sections), as.integer(mode),
        as.integer(n_ctx_orig), as.numeric(freq_base), as.numeric(freq_scale),
        as.numeric(ext_factor), as.numeric(attn_factor),
        as.numeric(beta_fast), as.numeric(beta_slow),
        PACKAGE = "ggmlR")
}

#' Multi-RoPE Inplace (Graph)
#'
#' Creates a graph node for multi-dimensional RoPE, modifying input in place.
#'
#' @inheritParams ggml_rope_multi
#' @return View of input tensor with MRoPE applied in place
#' @export
#' @family rope
ggml_rope_multi_inplace <- function(ctx, a, b, c = NULL,
                                     n_dims, sections = c(0L, 0L, 0L, 0L),
                                     mode = 0L, n_ctx_orig = 0L,
                                     freq_base = 10000.0, freq_scale = 1.0,
                                     ext_factor = 0.0, attn_factor = 1.0,
                                     beta_fast = 32.0, beta_slow = 1.0) {
  .Call("R_ggml_rope_multi_inplace", ctx, a, b, c,
        as.integer(n_dims), as.integer(sections), as.integer(mode),
        as.integer(n_ctx_orig), as.numeric(freq_base), as.numeric(freq_scale),
        as.numeric(ext_factor), as.numeric(attn_factor),
        as.numeric(beta_fast), as.numeric(beta_slow),
        PACKAGE = "ggmlR")
}

# ============================================================================
# Flash Attention
# ============================================================================

#' Flash Attention (Graph)
#'
#' Creates a graph node for Flash Attention computation.
#' This is a memory-efficient implementation of scaled dot-product attention.
#'
#' @param ctx GGML context
#' @param q Query tensor of shape [head_dim, n_head, n_tokens, batch]
#' @param k Key tensor of shape [head_dim, n_head_kv, n_kv, batch]
#' @param v Value tensor of shape [head_dim, n_head_kv, n_kv, batch]
#' @param mask Optional attention mask tensor (NULL for no mask).
#'   For causal attention, use ggml_diag_mask_inf instead.
#' @param scale Attention scale factor, typically 1/sqrt(head_dim)
#' @param max_bias Maximum ALiBi bias (0.0 to disable ALiBi)
#' @param logit_softcap Logit soft-capping value (0.0 to disable).
#'   Used by some models like Gemma 2.
#' @return Attention output tensor of shape [head_dim, n_head, n_tokens, batch]
#'
#' @details
#' Flash Attention computes: softmax(Q * K^T / scale + mask) * V
#'
#' Key features:
#' - Memory efficient: O(n) instead of O(n^2) memory for attention matrix
#' - Supports grouped-query attention (GQA) when n_head_kv < n_head
#' - Supports multi-query attention (MQA) when n_head_kv = 1
#' - Optional ALiBi (Attention with Linear Biases) for position encoding
#' - Optional logit soft-capping for numerical stability
#'
#' @examples
#' \donttest{
#' ctx <- ggml_init(64 * 1024 * 1024)
#' head_dim <- 64
#' n_head <- 8
#' n_head_kv <- 2  # GQA with 4:1 ratio
#' seq_len <- 32
#' q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
#' k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)
#' v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)
#' ggml_set_f32(q, rnorm(head_dim * n_head * seq_len))
#' ggml_set_f32(k, rnorm(head_dim * n_head_kv * seq_len))
#' ggml_set_f32(v, rnorm(head_dim * n_head_kv * seq_len))
#' # Scale = 1/sqrt(head_dim)
#' scale <- 1.0 / sqrt(head_dim)
#' # Compute attention
#' out <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0, 0.0)
#' graph <- ggml_build_forward_expand(ctx, out)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
#' @export
ggml_flash_attn_ext <- function(ctx, q, k, v, mask = NULL,
                                scale, max_bias = 0.0, logit_softcap = 0.0) {
  .Call("R_ggml_flash_attn_ext", ctx, q, k, v, mask,
        as.numeric(scale), as.numeric(max_bias), as.numeric(logit_softcap),
        PACKAGE = "ggmlR")
}

# ============================================================================
# View Operations with Offset
# ============================================================================

#' 1D View with Byte Offset (Graph)
#'
#' Creates a 1D view of a tensor starting at a byte offset.
#' The view shares memory with the source tensor.
#'
#' @param ctx GGML context
#' @param a Source tensor
#' @param ne0 Number of elements in the view
#' @param offset Byte offset from the start of tensor data
#' @return View tensor
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' # View elements 10-19 (offset = 10 * 4 bytes = 40)
#' v <- ggml_view_1d(ctx, a, 10, 40)
#' ggml_free(ctx)
#' }
ggml_view_1d <- function(ctx, a, ne0, offset = 0) {
  .Call("R_ggml_view_1d", ctx, a, as.numeric(ne0), as.numeric(offset), PACKAGE = "ggmlR")
}

#' 2D View with Byte Offset (Graph)
#'
#' Creates a 2D view of a tensor starting at a byte offset.
#' The view shares memory with the source tensor.
#'
#' @param ctx GGML context
#' @param a Source tensor
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @param nb1 Stride for dimension 1 (in bytes)
#' @param offset Byte offset from the start of tensor data
#' @return View tensor
#' @export
ggml_view_2d <- function(ctx, a, ne0, ne1, nb1, offset = 0) {
  .Call("R_ggml_view_2d", ctx, a, as.numeric(ne0), as.numeric(ne1),
        as.numeric(nb1), as.numeric(offset), PACKAGE = "ggmlR")
}

#' 3D View with Byte Offset (Graph)
#'
#' Creates a 3D view of a tensor starting at a byte offset.
#' The view shares memory with the source tensor.
#'
#' @param ctx GGML context
#' @param a Source tensor
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @param ne2 Size of dimension 2
#' @param nb1 Stride for dimension 1 (in bytes)
#' @param nb2 Stride for dimension 2 (in bytes)
#' @param offset Byte offset from the start of tensor data
#' @return View tensor
#' @export
ggml_view_3d <- function(ctx, a, ne0, ne1, ne2, nb1, nb2, offset = 0) {
  .Call("R_ggml_view_3d", ctx, a, as.numeric(ne0), as.numeric(ne1), as.numeric(ne2),
        as.numeric(nb1), as.numeric(nb2), as.numeric(offset), PACKAGE = "ggmlR")
}

#' 4D View with Byte Offset (Graph)
#'
#' Creates a 4D view of a tensor starting at a byte offset.
#' The view shares memory with the source tensor.
#' CRITICAL for KV-cache operations in transformers.
#'
#' @param ctx GGML context
#' @param a Source tensor
#' @param ne0 Size of dimension 0
#' @param ne1 Size of dimension 1
#' @param ne2 Size of dimension 2
#' @param ne3 Size of dimension 3
#' @param nb1 Stride for dimension 1 (in bytes)
#' @param nb2 Stride for dimension 2 (in bytes)
#' @param nb3 Stride for dimension 3 (in bytes)
#' @param offset Byte offset from the start of tensor data
#' @return View tensor
#' @export
ggml_view_4d <- function(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset = 0) {
  .Call("R_ggml_view_4d", ctx, a, as.numeric(ne0), as.numeric(ne1),
        as.numeric(ne2), as.numeric(ne3),
        as.numeric(nb1), as.numeric(nb2), as.numeric(nb3),
        as.numeric(offset), PACKAGE = "ggmlR")
}

# ============================================================================
# Copy Operation
# ============================================================================

#' Copy Tensor with Type Conversion (Graph)
#'
#' Copies tensor a into tensor b, performing type conversion if needed.
#' The tensors must have the same number of elements.
#' CRITICAL for type casting operations (e.g., F32 to F16).
#'
#' @param ctx GGML context
#' @param a Source tensor
#' @param b Destination tensor (defines output type and shape)
#' @return Tensor representing the copy operation (returns b with a's data)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Create F32 tensor
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' ggml_set_f32(a, rnorm(100))
#' # Create F16 tensor for output
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 100)
#' # Copy with F32 -> F16 conversion
#' result <- ggml_cpy(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, result)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_cpy <- function(ctx, a, b) {
  .Call("R_ggml_cpy", ctx, a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# Set Operations
# ============================================================================

#' Set Tensor Region (Graph)
#'
#' Copies tensor b into tensor a at a specified offset.
#' This allows writing to a portion of a tensor.
#'
#' @param ctx GGML context
#' @param a Destination tensor
#' @param b Source tensor (data to copy)
#' @param nb1 Stride for dimension 1 (in bytes)
#' @param nb2 Stride for dimension 2 (in bytes)
#' @param nb3 Stride for dimension 3 (in bytes)
#' @param offset Byte offset in destination tensor
#' @return Tensor representing the set operation
#' @export
ggml_set <- function(ctx, a, b, nb1, nb2, nb3, offset) {
  .Call("R_ggml_set", ctx, a, b, as.numeric(nb1), as.numeric(nb2),
        as.numeric(nb3), as.numeric(offset), PACKAGE = "ggmlR")
}

#' Set 1D Tensor Region (Graph)
#'
#' Simplified 1D version of ggml_set.
#' Copies tensor b into tensor a starting at offset.
#'
#' @param ctx GGML context
#' @param a Destination tensor
#' @param b Source tensor
#' @param offset Byte offset in destination tensor
#' @return Tensor representing the set operation
#' @export
ggml_set_1d <- function(ctx, a, b, offset) {
  .Call("R_ggml_set_1d", ctx, a, b, as.numeric(offset), PACKAGE = "ggmlR")
}

#' Set 2D Tensor Region (Graph)
#'
#' Simplified 2D version of ggml_set.
#'
#' @param ctx GGML context
#' @param a Destination tensor
#' @param b Source tensor
#' @param nb1 Stride for dimension 1 (in bytes)
#' @param offset Byte offset in destination tensor
#' @return Tensor representing the set operation
#' @export
ggml_set_2d <- function(ctx, a, b, nb1, offset) {
  .Call("R_ggml_set_2d", ctx, a, b, as.numeric(nb1), as.numeric(offset), PACKAGE = "ggmlR")
}

# ============================================================================
# Matrix Operations - Extended
# ============================================================================

#' Outer Product (Graph)
#'
#' Computes the outer product of two vectors: C = a * b^T
#' For vectors a[m] and b[n], produces matrix C[m, n].
#'
#' @param ctx GGML context
#' @param a First vector tensor
#' @param b Second vector tensor
#' @return Matrix tensor representing the outer product
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' ggml_set_f32(a, c(1, 2, 3))
#' ggml_set_f32(b, c(1, 2, 3, 4))
#' c <- ggml_out_prod(ctx, a, b)  # Result: 3x4 matrix
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_out_prod <- function(ctx, a, b) {
  .Call("R_ggml_out_prod", ctx, a, b, PACKAGE = "ggmlR")
}

#' Diagonal Matrix (Graph)
#'
#' Creates a diagonal matrix from a vector.
#' For vector a[n], produces matrix with a on the diagonal.
#'
#' @param ctx GGML context
#' @param a Input vector tensor
#' @return Diagonal matrix tensor
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
#' ggml_set_f32(a, c(1, 2, 3))
#' d <- ggml_diag(ctx, a)  # 3x3 diagonal matrix
#' graph <- ggml_build_forward_expand(ctx, d)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_diag <- function(ctx, a) {
  .Call("R_ggml_diag", ctx, a, PACKAGE = "ggmlR")
}

# ============================================================================
# Concatenation
# ============================================================================

#' Concatenate Tensors (Graph)
#'
#' Concatenates two tensors along a specified dimension.
#' CRITICAL for KV-cache operations in transformers.
#'
#' @param ctx GGML context
#' @param a First tensor
#' @param b Second tensor (must match a in all dimensions except the concat dim)
#' @param dim Dimension along which to concatenate (0-3)
#' @return Concatenated tensor
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3)
#' b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2)
#' ggml_set_f32(a, rnorm(12))
#' ggml_set_f32(b, rnorm(8))
#' # Concatenate along dimension 1: result is 4x5
#' c <- ggml_concat(ctx, a, b, 1)
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_concat <- function(ctx, a, b, dim = 0) {
  .Call("R_ggml_concat", ctx, a, b, as.integer(dim), PACKAGE = "ggmlR")
}

# ============================================================================
# Backward Pass Operations (for Training)
# ============================================================================

#' SiLU Backward (Graph)
#'
#' Computes the backward pass for SiLU (Swish) activation.
#' Used during training for gradient computation.
#'
#' @param ctx GGML context
#' @param a Forward input tensor
#' @param b Gradient tensor from upstream
#' @return Gradient tensor for the input
#' @export
ggml_silu_back <- function(ctx, a, b) {
  .Call("R_ggml_silu_back", ctx, a, b, PACKAGE = "ggmlR")
}

#' Get Rows Backward (Graph)
#'
#' Backward pass for ggml_get_rows operation.
#' Accumulates gradients at the original row positions.
#'
#' @param ctx GGML context
#' @param a Gradient of get_rows output
#' @param b Index tensor (same as forward pass)
#' @param c Reference tensor defining output shape
#' @return Gradient tensor for the embedding matrix
#' @export
ggml_get_rows_back <- function(ctx, a, b, c) {
  .Call("R_ggml_get_rows_back", ctx, a, b, c, PACKAGE = "ggmlR")
}

#' Softmax Backward Extended (Graph)
#'
#' Backward pass for extended softmax operation.
#'
#' @param ctx GGML context
#' @param a Softmax output tensor (from forward pass)
#' @param b Gradient tensor from upstream
#' @param scale Scale factor (same as forward pass)
#' @param max_bias Maximum ALiBi bias (same as forward pass)
#' @return Gradient tensor for the input
#' @export
ggml_soft_max_ext_back <- function(ctx, a, b, scale = 1.0, max_bias = 0.0) {
  .Call("R_ggml_soft_max_ext_back", ctx, a, b,
        as.numeric(scale), as.numeric(max_bias), PACKAGE = "ggmlR")
}

#' RoPE Extended Backward (Graph)
#'
#' Backward pass for extended RoPE (Rotary Position Embedding).
#' Used during training to compute gradients through RoPE.
#'
#' @param ctx GGML context
#' @param a Gradient tensor from upstream (gradients of ggml_rope_ext result)
#' @param b Position tensor (same as forward pass)
#' @param c Optional frequency factors tensor (NULL for default)
#' @param n_dims Number of dimensions for rotation
#' @param mode RoPE mode
#' @param n_ctx_orig Original context length
#' @param freq_base Base frequency
#' @param freq_scale Frequency scale factor
#' @param ext_factor Extension factor (YaRN)
#' @param attn_factor Attention factor
#' @param beta_fast YaRN fast beta
#' @param beta_slow YaRN slow beta
#' @return Gradient tensor for the input
#' @export
ggml_rope_ext_back <- function(ctx, a, b, c = NULL,
                               n_dims, mode = 0L, n_ctx_orig = 0L,
                               freq_base = 10000.0, freq_scale = 1.0,
                               ext_factor = 0.0, attn_factor = 1.0,
                               beta_fast = 32.0, beta_slow = 1.0) {
  .Call("R_ggml_rope_ext_back", ctx, a, b, c,
        as.integer(n_dims), as.integer(mode), as.integer(n_ctx_orig),
        as.numeric(freq_base), as.numeric(freq_scale),
        as.numeric(ext_factor), as.numeric(attn_factor),
        as.numeric(beta_fast), as.numeric(beta_slow),
        PACKAGE = "ggmlR")
}

#' Flash Attention Backward (Graph)
#'
#' Backward pass for Flash Attention.
#' Used during training to compute gradients through attention.
#'
#' @param ctx GGML context
#' @param q Query tensor (same as forward pass)
#' @param k Key tensor (same as forward pass)
#' @param v Value tensor (same as forward pass)
#' @param d Gradient tensor from upstream (same shape as forward output)
#' @param masked Logical: whether causal masking was used in forward pass
#' @return Gradient tensor
#' @export
ggml_flash_attn_back <- function(ctx, q, k, v, d, masked = TRUE) {
  .Call("R_ggml_flash_attn_back", ctx, q, k, v, d, as.logical(masked),
        PACKAGE = "ggmlR")
}

# ============================================================================
# Mixture of Experts (MoE) Operations
# ============================================================================

#' Matrix Multiplication with Expert Selection (Graph)
#'
#' Indirect matrix multiplication for Mixture of Experts architectures.
#' Selects expert weights based on indices and performs batched matmul.
#'
#' @param ctx GGML context
#' @param as Stacked expert weight matrices [n_embd, n_ff, n_experts]
#' @param b Input tensor
#' @param ids Expert selection indices tensor (I32)
#' @return Output tensor after expert-selected matrix multiplication
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(64 * 1024 * 1024)
#' # 4 experts, each with 8x16 weights (small for example)
#' experts <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 16, 4)
#' ggml_set_f32(experts, rnorm(8 * 16 * 4))
#' input <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2)
#' ggml_set_f32(input, rnorm(16))
#' # Select expert 0 for token 0, expert 2 for token 1
#' ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2)
#' ggml_set_i32(ids, c(0L, 2L))
#' output <- ggml_mul_mat_id(ctx, experts, input, ids)
#' graph <- ggml_build_forward_expand(ctx, output)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_mul_mat_id <- function(ctx, as, b, ids) {
  .Call("R_ggml_mul_mat_id", ctx, as, b, ids, PACKAGE = "ggmlR")
}

# ============================================================================
# Sequence/Token Operations
# ============================================================================

#' Pad Tensor with Zeros (Graph)
#'
#' Pads tensor dimensions with zeros on the right side.
#' Useful for aligning tensor sizes in attention operations.
#'
#' @param ctx GGML context
#' @param a Input tensor to pad
#' @param p0 Padding for dimension 0 (default 0)
#' @param p1 Padding for dimension 1 (default 0)
#' @param p2 Padding for dimension 2 (default 0)
#' @param p3 Padding for dimension 3 (default 0)
#' @return Padded tensor with shape [ne0+p0, ne1+p1, ne2+p2, ne3+p3]
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 3)
#' ggml_set_f32(a, 1:15)
#' # Pad to 8x4
#' b <- ggml_pad(ctx, a, 3, 1)  # Add 3 zeros to dim0, 1 to dim1
#' graph <- ggml_build_forward_expand(ctx, b)
#' ggml_graph_compute(ctx, graph)
#' # Result shape: [8, 4]
#' ggml_free(ctx)
#' }
ggml_pad <- function(ctx, a, p0 = 0L, p1 = 0L, p2 = 0L, p3 = 0L) {
  .Call("R_ggml_pad", ctx, a, as.integer(p0), as.integer(p1),
        as.integer(p2), as.integer(p3), PACKAGE = "ggmlR")
}

#' Sort Order Constants
#'
#' Sort Order Constants
#'
#' Constants for specifying sort order in argsort operations.
#'
#' @format Integer constants
#' @return An integer constant representing a sort order
#' @details
#' \itemize{
#'   \item \code{GGML_SORT_ORDER_ASC} (0): Ascending order (smallest first)
#'   \item \code{GGML_SORT_ORDER_DESC} (1): Descending order (largest first)
#' }
#' @examples
#' \donttest{
#' GGML_SORT_ORDER_ASC   # 0 - Ascending order
#' GGML_SORT_ORDER_DESC  # 1 - Descending order
#'
#' # Usage with ggml_argsort
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(3, 1, 4, 1, 5))
#' # Get ascending sort indices
#' idx_asc <- ggml_argsort(ctx, a, GGML_SORT_ORDER_ASC)
#' # Get descending sort indices
#' idx_desc <- ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC)
#' ggml_free(ctx)
#' }
#' @export
GGML_SORT_ORDER_ASC <- 0L

#' @rdname GGML_SORT_ORDER_ASC
#' @export
GGML_SORT_ORDER_DESC <- 1L

#' Argsort - Get Sorting Indices (Graph)
#'
#' Returns indices that would sort the tensor rows.
#' Each row is sorted independently.
#'
#' @param ctx GGML context
#' @param a Input tensor to sort (F32)
#' @param order Sort order: GGML_SORT_ORDER_ASC (0) or GGML_SORT_ORDER_DESC (1)
#' @return Tensor of I32 indices that would sort each row
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Create tensor with values to sort
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(3, 1, 4, 1, 5))
#' # Get indices for ascending sort
#' indices <- ggml_argsort(ctx, a, GGML_SORT_ORDER_ASC)
#' graph <- ggml_build_forward_expand(ctx, indices)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_i32(indices)
#' # result: [1, 3, 0, 2, 4] (0-indexed positions for sorted order)
#' ggml_free(ctx)
#' }
ggml_argsort <- function(ctx, a, order = GGML_SORT_ORDER_ASC) {
  .Call("R_ggml_argsort", ctx, a, as.integer(order), PACKAGE = "ggmlR")
}

#' Top-K Indices (Graph)
#'
#' Returns the indices of top K elements per row.
#' Useful for sampling strategies in language models (top-k sampling).
#' Note: the resulting indices are in no particular order within top-k.
#'
#' @param ctx GGML context
#' @param a Input tensor (F32)
#' @param k Number of top elements to return per row
#' @return Tensor containing I32 indices of top-k elements (not values)
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Logits from model output
#' logits <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100)
#' ggml_set_f32(logits, rnorm(100))
#' # Get top 5 logits for sampling
#' top5 <- ggml_top_k(ctx, logits, 5)
#' graph <- ggml_build_forward_expand(ctx, top5)
#' ggml_graph_compute(ctx, graph)
#' ggml_free(ctx)
#' }
ggml_top_k <- function(ctx, a, k) {
  .Call("R_ggml_top_k", ctx, a, as.integer(k), PACKAGE = "ggmlR")
}

# ============================================================================
# Additional Sequence Operations
# ============================================================================

#' Repeat Backward (Graph)
#'
#' Backward pass for repeat operation - sums repetitions back to original shape.
#' Used for gradient computation during training.
#'
#' @param ctx GGML context
#' @param a Input tensor (gradients from repeated tensor)
#' @param b Target shape tensor (original tensor before repeat)
#' @return Tensor with summed gradients matching shape of b
#' @export
ggml_repeat_back <- function(ctx, a, b) {
  .Call("R_ggml_repeat_back", ctx, a, b, PACKAGE = "ggmlR")
}

#' Upscale Tensor (Graph)
#'
#' Upscales tensor by multiplying ne0 and ne1 by scale factor.
#' Supports different interpolation modes for image upscaling.
#'
#' @param ctx GGML context
#' @param a Input tensor (typically 2D or 4D for images)
#' @param scale_factor Integer scale factor (e.g., 2 = double size)
#' @param mode Scale mode constant (see details)
#' @return Upscaled tensor with dimensions multiplied by scale_factor
#'
#' @details
#' Scale mode constants:
#' \itemize{
#'   \item \code{GGML_SCALE_MODE_NEAREST} (0): Nearest neighbor interpolation - fastest, pixelated
#'   \item \code{GGML_SCALE_MODE_BILINEAR} (1): Bilinear interpolation - smooth, good balance
#'   \item \code{GGML_SCALE_MODE_BICUBIC} (2): Bicubic interpolation - smoothest, most compute
#' }
#'
#' @export
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' img <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8)
#' ggml_set_f32(img, rnorm(64))
#'
#' # Nearest neighbor (fastest, pixelated)
#' up_nearest <- ggml_upscale(ctx, img, 2, GGML_SCALE_MODE_NEAREST)
#'
#' # Bilinear (smooth)
#' up_bilinear <- ggml_upscale(ctx, img, 2, GGML_SCALE_MODE_BILINEAR)
#'
#' # Bicubic (smoothest)
#' up_bicubic <- ggml_upscale(ctx, img, 2, GGML_SCALE_MODE_BICUBIC)
#'
#' graph <- ggml_build_forward_expand(ctx, up_nearest)
#' ggml_graph_compute(ctx, graph)
#' # Result is 16x16
#' ggml_free(ctx)
#' }
ggml_upscale <- function(ctx, a, scale_factor, mode = 0L) {
  .Call("R_ggml_upscale", ctx, a, as.integer(scale_factor), as.integer(mode),
        PACKAGE = "ggmlR")
}

#' @rdname ggml_upscale
#' @export
GGML_SCALE_MODE_NEAREST <- 0L

#' @rdname ggml_upscale
#' @export
GGML_SCALE_MODE_BILINEAR <- 1L

#' @rdname ggml_upscale
#' @export
GGML_SCALE_MODE_BICUBIC <- 2L

# ============================================================================
# Additional Utility Functions
# ============================================================================

#' Get Type Size in Bytes
#'
#' Returns the size in bytes for all elements in a block for a given type.
#'
#' @param type GGML type constant (e.g., GGML_TYPE_F32)
#' @return Size in bytes
#' @export
ggml_type_size <- function(type) {
  .Call("R_ggml_type_size", as.integer(type), PACKAGE = "ggmlR")
}

#' Get Element Size
#'
#' Returns the size of a single element in the tensor.
#'
#' @param tensor Tensor pointer
#' @return Element size in bytes
#' @export
ggml_element_size <- function(tensor) {
  .Call("R_ggml_element_size", tensor, PACKAGE = "ggmlR")
}

#' Get Number of Rows
#'
#' Returns the number of rows in a tensor (product of all dimensions except ne[0]).
#'
#' @param tensor Tensor pointer
#' @return Number of rows
#' @export
ggml_nrows <- function(tensor) {
  .Call("R_ggml_nrows", tensor, PACKAGE = "ggmlR")
}

#' Compare Tensor Shapes
#'
#' Checks if two tensors have the same shape.
#'
#' @param a First tensor
#' @param b Second tensor
#' @return TRUE if shapes are identical, FALSE otherwise
#' @export
ggml_are_same_shape <- function(a, b) {
  .Call("R_ggml_are_same_shape", a, b, PACKAGE = "ggmlR")
}

#' Set Tensor Name
#'
#' Assigns a name to a tensor. Useful for debugging and graph visualization.
#'
#' @param tensor Tensor pointer
#' @param name Character string name
#' @return The tensor (for chaining)
#' @export
ggml_set_name <- function(tensor, name) {
  .Call("R_ggml_set_name", tensor, as.character(name), PACKAGE = "ggmlR")
}

#' Set Tensor as Trainable Parameter
#'
#' Marks a tensor as a trainable parameter for backpropagation.
#' The optimizer will compute gradients for this tensor during training.
#'
#' @param tensor Tensor pointer
#' @return The tensor (for chaining)
#' @export
ggml_set_param <- function(tensor) {
  .Call("R_ggml_set_param", tensor, PACKAGE = "ggmlR")
}

#' Mark Tensor as Input
#'
#' @param tensor Tensor pointer
#' @return The tensor (for chaining)
#' @export
ggml_set_input <- function(tensor) {
  .Call("R_ggml_set_input", tensor, PACKAGE = "ggmlR")
}

#' Mark Tensor as Output
#'
#' @param tensor Tensor pointer
#' @return The tensor (for chaining)
#' @export
ggml_set_output <- function(tensor) {
  .Call("R_ggml_set_output", tensor, PACKAGE = "ggmlR")
}

#' Get Tensor Name
#'
#' Retrieves the name of a tensor.
#'
#' @param tensor Tensor pointer
#' @return Character string name or NULL if not set
#' @export
ggml_get_name <- function(tensor) {
  .Call("R_ggml_get_name", tensor, PACKAGE = "ggmlR")
}

# ============================================================================
# Backend Functions - Direct Access
# ============================================================================

#' Initialize CPU Backend
#'
#' Creates a new CPU backend instance for graph computation.
#'
#' @return Backend pointer
#' @export
ggml_backend_cpu_init <- function() {
  .Call("R_ggml_backend_cpu_init", PACKAGE = "ggmlR")
}

#' Free Backend
#'
#' Releases resources associated with a backend.
#'
#' @param backend Backend pointer
#' @return NULL invisibly
#' @export
ggml_backend_free <- function(backend) {
  invisible(.Call("R_ggml_backend_free", backend, PACKAGE = "ggmlR"))
}

#' Set CPU Backend Threads
#'
#' Sets the number of threads for CPU backend computation.
#'
#' @param backend CPU backend pointer
#' @param n_threads Number of threads
#' @return NULL invisibly
#' @export
ggml_backend_cpu_set_n_threads <- function(backend, n_threads) {
  invisible(.Call("R_ggml_backend_cpu_set_n_threads", backend,
                  as.integer(n_threads), PACKAGE = "ggmlR"))
}

#' Compute Graph with Backend
#'
#' Executes computation graph using specified backend.
#'
#' @param backend Backend pointer
#' @param graph Graph pointer
#' @return Status code (0 = success)
#' @export
ggml_backend_graph_compute <- function(backend, graph) {
  .Call("R_ggml_backend_graph_compute", backend, graph, PACKAGE = "ggmlR")
}

#' Get Backend Name
#'
#' Returns the name of the backend (e.g., "CPU").
#'
#' @param backend Backend pointer
#' @return Character string name
#' @export
ggml_backend_name <- function(backend) {
  .Call("R_ggml_backend_name", backend, PACKAGE = "ggmlR")
}

# ============================================================================
# CNN Operations
# ============================================================================

#' 1D Convolution (Graph)
#'
#' Applies 1D convolution to input data.
#'
#' @param ctx GGML context
#' @param a Convolution kernel tensor
#' @param b Input data tensor
#' @param s0 Stride (default 1)
#' @param p0 Padding (default 0)
#' @param d0 Dilation (default 1)
#' @return Convolved tensor
#' @export
ggml_conv_1d <- function(ctx, a, b, s0 = 1L, p0 = 0L, d0 = 1L) {
  .Call("R_ggml_conv_1d", ctx, a, b,
        as.integer(s0), as.integer(p0), as.integer(d0),
        PACKAGE = "ggmlR")
}

#' 2D Convolution (Graph)
#'
#' Applies 2D convolution to input data.
#'
#' @param ctx GGML context
#' @param a Convolution kernel tensor [KW, KH, IC, OC]
#' @param b Input data tensor [W, H, C, N]
#' @param s0 Stride dimension 0 (default 1)
#' @param s1 Stride dimension 1 (default 1)
#' @param p0 Padding dimension 0 (default 0)
#' @param p1 Padding dimension 1 (default 0)
#' @param d0 Dilation dimension 0 (default 1)
#' @param d1 Dilation dimension 1 (default 1)
#' @return Convolved tensor
#' @export
ggml_conv_2d <- function(ctx, a, b, s0 = 1L, s1 = 1L, p0 = 0L, p1 = 0L, d0 = 1L, d1 = 1L) {
  .Call("R_ggml_conv_2d", ctx, a, b,
        as.integer(s0), as.integer(s1),
        as.integer(p0), as.integer(p1),
        as.integer(d0), as.integer(d1),
        PACKAGE = "ggmlR")
}

#' Transposed 1D Convolution (Graph)
#'
#' Applies transposed 1D convolution (deconvolution) to input data.
#'
#' @param ctx GGML context
#' @param a Convolution kernel tensor
#' @param b Input data tensor
#' @param s0 Stride (default 1)
#' @param p0 Padding (default 0)
#' @param d0 Dilation (default 1)
#' @return Transposed convolved tensor
#' @export
ggml_conv_transpose_1d <- function(ctx, a, b, s0 = 1L, p0 = 0L, d0 = 1L) {
  .Call("R_ggml_conv_transpose_1d", ctx, a, b,
        as.integer(s0), as.integer(p0), as.integer(d0),
        PACKAGE = "ggmlR")
}

#' 1D Pooling (Graph)
#'
#' Applies 1D pooling operation for downsampling.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param op Pool operation constant (see details)
#' @param k0 Kernel size (window size)
#' @param s0 Stride (default = k0 for non-overlapping windows)
#' @param p0 Padding (default 0)
#' @return Pooled tensor with reduced dimensions
#'
#' @details
#' Pool operation constants:
#' \itemize{
#'   \item \code{GGML_OP_POOL_MAX} (0): Max pooling - takes maximum value in each window
#'   \item \code{GGML_OP_POOL_AVG} (1): Average pooling - takes mean of values in each window
#' }
#'
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8)
#' ggml_set_f32(a, c(1, 3, 2, 4, 5, 2, 8, 1))
#'
#' # Max pooling with kernel 2, stride 2
#' max_pool <- ggml_pool_1d(ctx, a, GGML_OP_POOL_MAX, k0 = 2)
#' # Result: [3, 4, 5, 8] (max of each pair)
#'
#' # Average pooling with kernel 2, stride 2
#' avg_pool <- ggml_pool_1d(ctx, a, GGML_OP_POOL_AVG, k0 = 2)
#' # Result: [2, 3, 3.5, 4.5] (mean of each pair)
#'
#' ggml_free(ctx)
#' }
#' @export
ggml_pool_1d <- function(ctx, a, op, k0, s0 = k0, p0 = 0L) {
  .Call("R_ggml_pool_1d", ctx, a, as.integer(op),
        as.integer(k0), as.integer(s0), as.integer(p0),
        PACKAGE = "ggmlR")
}

#' 2D Pooling (Graph)
#'
#' Applies 2D pooling operation.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param op Pool operation: GGML_OP_POOL_MAX (0) or GGML_OP_POOL_AVG (1)
#' @param k0 Kernel size dimension 0
#' @param k1 Kernel size dimension 1
#' @param s0 Stride dimension 0 (default = k0)
#' @param s1 Stride dimension 1 (default = k1)
#' @param p0 Padding dimension 0 (default 0)
#' @param p1 Padding dimension 1 (default 0)
#' @return Pooled tensor
#' @export
ggml_pool_2d <- function(ctx, a, op, k0, k1, s0 = k0, s1 = k1, p0 = 0, p1 = 0) {
  .Call("R_ggml_pool_2d", ctx, a, as.integer(op),
        as.integer(k0), as.integer(k1),
        as.integer(s0), as.integer(s1),
        as.numeric(p0), as.numeric(p1),
        PACKAGE = "ggmlR")
}

#' Image to Column (Graph)
#'
#' Transforms image data into column format for efficient convolution.
#' This is a low-level operation used internally by convolution implementations.
#'
#' @param ctx GGML context
#' @param a Convolution kernel tensor
#' @param b Input data tensor
#' @param s0 Stride dimension 0
#' @param s1 Stride dimension 1
#' @param p0 Padding dimension 0
#' @param p1 Padding dimension 1
#' @param d0 Dilation dimension 0
#' @param d1 Dilation dimension 1
#' @param is_2D Whether this is a 2D operation (default TRUE)
#' @param dst_type Output type (default GGML_TYPE_F16)
#' @return Transformed tensor in column format
#' @export
ggml_im2col <- function(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D = TRUE, dst_type = GGML_TYPE_F16) {
  .Call("R_ggml_im2col", ctx, a, b,
        as.integer(s0), as.integer(s1),
        as.integer(p0), as.integer(p1),
        as.integer(d0), as.integer(d1),
        as.logical(is_2D),
        as.integer(dst_type),
        PACKAGE = "ggmlR")
}

#' @rdname ggml_pool_1d
#' @export
GGML_OP_POOL_MAX <- 0L

#' @rdname ggml_pool_1d
#' @export
GGML_OP_POOL_AVG <- 1L

# ============================================================================
# Quantization Functions
# ============================================================================

#' Initialize Quantization Tables
#'
#' Initializes quantization tables for a given type.
#' Called automatically by ggml_quantize_chunk, but can be called manually.
#'
#' @param type GGML type (e.g., GGML_TYPE_Q4_0)
#' @return NULL invisibly
#' @export
ggml_quantize_init <- function(type) {
  invisible(.Call("R_ggml_quantize_init", as.integer(type), PACKAGE = "ggmlR"))
}

#' Free Quantization Resources
#'
#' Frees any memory allocated by quantization.
#' Call at end of program to avoid memory leaks.
#'
#' @return NULL invisibly
#' @export
ggml_quantize_free <- function() {
  invisible(.Call("R_ggml_quantize_free", PACKAGE = "ggmlR"))
}

#' Check if Quantization Requires Importance Matrix
#'
#' Some quantization types require an importance matrix for optimal quality.
#'
#' @param type GGML type
#' @return TRUE if importance matrix is required
#' @export
ggml_quantize_requires_imatrix <- function(type) {
  .Call("R_ggml_quantize_requires_imatrix", as.integer(type), PACKAGE = "ggmlR")
}

#' Quantize Data Chunk
#'
#' Quantizes a chunk of floating-point data to a lower precision format.
#'
#' @param type Target GGML type (e.g., GGML_TYPE_Q4_0)
#' @param src Source numeric vector (F32 data)
#' @param nrows Number of rows
#' @param n_per_row Number of elements per row
#' @return Raw vector containing quantized data
#' @export
#' @examples
#' \donttest{
#' # Quantize 256 floats to Q8_0 (block size 32)
#' data <- rnorm(256)
#' quantized <- ggml_quantize_chunk(GGML_TYPE_Q8_0, data, 1, 256)
#' ggml_quantize_free()  # Clean up
#' }
ggml_quantize_chunk <- function(type, src, nrows, n_per_row) {
  .Call("R_ggml_quantize_chunk", as.integer(type), as.numeric(src),
        as.numeric(nrows), as.numeric(n_per_row), PACKAGE = "ggmlR")
}

# ============================================================================
# Type System Functions
# ============================================================================

#' Get Type Name
#'
#' Returns the string name of a GGML type.
#'
#' @param type GGML type constant (e.g., GGML_TYPE_F32)
#' @return Character string with type name
#' @export
#' @family type_system
#' @examples
#' ggml_type_name(GGML_TYPE_F32)  # "f32"
#' ggml_type_name(GGML_TYPE_Q4_0) # "q4_0"
ggml_type_name <- function(type) {
  .Call("R_ggml_type_name", as.integer(type), PACKAGE = "ggmlR")
}

#' Get Type Size as Float
#'
#' Returns the size in bytes of a GGML type as a floating-point number.
#' For quantized types, this is the average bytes per element.
#'
#' @param type GGML type constant
#' @return Numeric size in bytes (can be fractional for quantized types)
#' @export
#' @family type_system
#' @examples
#' ggml_type_sizef(GGML_TYPE_F32)  # 4.0
#' ggml_type_sizef(GGML_TYPE_F16)  # 2.0
ggml_type_sizef <- function(type) {
  .Call("R_ggml_type_sizef", as.integer(type), PACKAGE = "ggmlR")
}

#' Get Block Size
#'
#' Returns the block size for a GGML type. Quantized types process
#' data in blocks (e.g., 32 elements for Q4_0).
#'
#' @param type GGML type constant
#' @return Integer block size
#' @export
#' @family type_system
#' @examples
#' ggml_blck_size(GGML_TYPE_F32)  # 1
#' ggml_blck_size(GGML_TYPE_Q4_0) # 32
ggml_blck_size <- function(type) {
  .Call("R_ggml_blck_size", as.integer(type), PACKAGE = "ggmlR")
}

#' Check If Type is Quantized
#'
#' Returns TRUE if the GGML type is a quantized format.
#'
#' @param type GGML type constant
#' @return Logical indicating if type is quantized
#' @export
#' @family type_system
#' @examples
#' ggml_is_quantized(GGML_TYPE_F32)  # FALSE
#' ggml_is_quantized(GGML_TYPE_Q4_0) # TRUE
ggml_is_quantized <- function(type) {
  .Call("R_ggml_is_quantized", as.integer(type), PACKAGE = "ggmlR")
}

#' Convert ftype to ggml_type
#'
#' Converts a file type (ftype) to the corresponding GGML type.
#' Used when loading quantized models.
#'
#' @param ftype File type constant
#' @return Integer GGML type
#' @export
#' @family type_system
ggml_ftype_to_ggml_type <- function(ftype) {
  .Call("R_ggml_ftype_to_ggml_type", as.integer(ftype), PACKAGE = "ggmlR")
}

# ============================================================================
# Operation Info Functions
# ============================================================================

#' Get Operation Name
#'
#' Returns the string name of a GGML operation.
#'
#' @param op GGML operation constant
#' @return Character string with operation name
#' @export
#' @family op_info
ggml_op_name <- function(op) {
  .Call("R_ggml_op_name", as.integer(op), PACKAGE = "ggmlR")
}

#' Get Operation Symbol
#'
#' Returns the mathematical symbol for a GGML operation.
#'
#' @param op GGML operation constant
#' @return Character string with operation symbol
#' @export
#' @family op_info
ggml_op_symbol <- function(op) {
  .Call("R_ggml_op_symbol", as.integer(op), PACKAGE = "ggmlR")
}

#' Get Unary Operation Name
#'
#' Returns the string name of a GGML unary operation.
#'
#' @param op GGML unary operation constant
#' @return Character string with operation name
#' @export
#' @family op_info
ggml_unary_op_name <- function(op) {
  .Call("R_ggml_unary_op_name", as.integer(op), PACKAGE = "ggmlR")
}

#' Get Operation Description from Tensor
#'
#' Returns a description of the operation that produces a tensor.
#'
#' @param tensor Tensor pointer
#' @return Character string describing the operation
#' @export
#' @family op_info
ggml_op_desc <- function(tensor) {
  .Call("R_ggml_op_desc", tensor, PACKAGE = "ggmlR")
}

#' Get Unary Operation from Tensor
#'
#' Returns the unary operation type for a unary operation tensor.
#'
#' @param tensor Tensor pointer (must be a unary operation result)
#' @return Integer unary operation type
#' @export
#' @family op_info
ggml_get_unary_op <- function(tensor) {
  .Call("R_ggml_get_unary_op", tensor, PACKAGE = "ggmlR")
}

# ============================================================================
# CPU Feature Detection Functions
# ============================================================================

#' CPU Feature Detection - SSE3
#'
#' Check if the CPU supports SSE3 instructions.
#'
#' @return Logical indicating SSE3 support
#' @export
#' @family cpu_features
#' @examples
#' ggml_cpu_has_sse3()
ggml_cpu_has_sse3 <- function() {
  .Call("R_ggml_cpu_has_sse3", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - SSSE3
#'
#' Check if the CPU supports SSSE3 instructions.
#'
#' @return Logical indicating SSSE3 support
#' @export
#' @family cpu_features
ggml_cpu_has_ssse3 <- function() {
  .Call("R_ggml_cpu_has_ssse3", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AVX
#'
#' Check if the CPU supports AVX instructions.
#'
#' @return Logical indicating AVX support
#' @export
#' @family cpu_features
ggml_cpu_has_avx <- function() {
  .Call("R_ggml_cpu_has_avx", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AVX-VNNI
#'
#' Check if the CPU supports AVX-VNNI instructions.
#'
#' @return Logical indicating AVX-VNNI support
#' @export
#' @family cpu_features
ggml_cpu_has_avx_vnni <- function() {
  .Call("R_ggml_cpu_has_avx_vnni", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AVX2
#'
#' Check if the CPU supports AVX2 instructions.
#' AVX2 provides 256-bit SIMD operations for faster matrix math.
#'
#' @return Logical indicating AVX2 support
#' @export
#' @family cpu_features
ggml_cpu_has_avx2 <- function() {
  .Call("R_ggml_cpu_has_avx2", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - BMI2
#'
#' Check if the CPU supports BMI2 (Bit Manipulation Instructions 2).
#'
#' @return Logical indicating BMI2 support
#' @export
#' @family cpu_features
ggml_cpu_has_bmi2 <- function() {
  .Call("R_ggml_cpu_has_bmi2", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - F16C
#'
#' Check if the CPU supports F16C instructions for float16 conversion.
#'
#' @return Logical indicating F16C support
#' @export
#' @family cpu_features
ggml_cpu_has_f16c <- function() {
  .Call("R_ggml_cpu_has_f16c", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - FMA
#'
#' Check if the CPU supports FMA (Fused Multiply-Add) instructions.
#' FMA allows matrix operations to run faster by combining operations.
#'
#' @return Logical indicating FMA support
#' @export
#' @family cpu_features
ggml_cpu_has_fma <- function() {
  .Call("R_ggml_cpu_has_fma", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AVX-512
#'
#' Check if the CPU supports AVX-512 instructions.
#' AVX-512 provides 512-bit SIMD for maximum throughput.
#'
#' @return Logical indicating AVX-512 support
#' @export
#' @family cpu_features
ggml_cpu_has_avx512 <- function() {
  .Call("R_ggml_cpu_has_avx512", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AVX-512 VBMI
#'
#' Check if the CPU supports AVX-512 VBMI instructions.
#'
#' @return Logical indicating AVX-512 VBMI support
#' @export
#' @family cpu_features
ggml_cpu_has_avx512_vbmi <- function() {
  .Call("R_ggml_cpu_has_avx512_vbmi", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AVX-512 VNNI
#'
#' Check if the CPU supports AVX-512 VNNI instructions.
#' VNNI accelerates neural network inference with int8/int16 dot products.
#'
#' @return Logical indicating AVX-512 VNNI support
#' @export
#' @family cpu_features
ggml_cpu_has_avx512_vnni <- function() {
  .Call("R_ggml_cpu_has_avx512_vnni", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AVX-512 BF16
#'
#' Check if the CPU supports AVX-512 BF16 (bfloat16) instructions.
#'
#' @return Logical indicating AVX-512 BF16 support
#' @export
#' @family cpu_features
ggml_cpu_has_avx512_bf16 <- function() {
  .Call("R_ggml_cpu_has_avx512_bf16", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - AMX INT8
#'
#' Check if the CPU supports AMX INT8 (Advanced Matrix Extensions).
#' AMX provides hardware acceleration for matrix operations on Intel CPUs.
#'
#' @return Logical indicating AMX INT8 support
#' @export
#' @family cpu_features
ggml_cpu_has_amx_int8 <- function() {
  .Call("R_ggml_cpu_has_amx_int8", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - NEON (ARM)
#'
#' Check if the CPU supports ARM NEON instructions.
#' NEON is ARM's SIMD extension for vectorized operations.
#'
#' @return Logical indicating NEON support
#' @export
#' @family cpu_features
ggml_cpu_has_neon <- function() {
  .Call("R_ggml_cpu_has_neon", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - ARM FMA
#'
#' Check if the CPU supports ARM FMA (Fused Multiply-Add).
#'
#' @return Logical indicating ARM FMA support
#' @export
#' @family cpu_features
ggml_cpu_has_arm_fma <- function() {
  .Call("R_ggml_cpu_has_arm_fma", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - FP16 Vector Arithmetic (ARM)
#'
#' Check if the CPU supports ARM half-precision FP16 vector arithmetic.
#'
#' @return Logical indicating FP16 VA support
#' @export
#' @family cpu_features
ggml_cpu_has_fp16_va <- function() {
  .Call("R_ggml_cpu_has_fp16_va", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - Dot Product (ARM)
#'
#' Check if the CPU supports ARM dot product instructions.
#' Accelerates int8 matrix multiplication common in quantized models.
#'
#' @return Logical indicating dot product support
#' @export
#' @family cpu_features
ggml_cpu_has_dotprod <- function() {
  .Call("R_ggml_cpu_has_dotprod", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - INT8 Matrix Multiply (ARM)
#'
#' Check if the CPU supports ARM INT8 matrix multiplication.
#'
#' @return Logical indicating INT8 MATMUL support
#' @export
#' @family cpu_features
ggml_cpu_has_matmul_int8 <- function() {
  .Call("R_ggml_cpu_has_matmul_int8", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - SVE (ARM)
#'
#' Check if the CPU supports ARM SVE (Scalable Vector Extension).
#'
#' @return Logical indicating SVE support
#' @export
#' @family cpu_features
ggml_cpu_has_sve <- function() {
  .Call("R_ggml_cpu_has_sve", PACKAGE = "ggmlR")
}

#' Get SVE Vector Length (ARM)
#'
#' Returns the SVE vector length in bytes (0 if not supported).
#'
#' @return Integer vector length in bytes
#' @export
#' @family cpu_features
ggml_cpu_get_sve_cnt <- function() {
  .Call("R_ggml_cpu_get_sve_cnt", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - SME (ARM)
#'
#' Check if the CPU supports ARM SME (Scalable Matrix Extension).
#'
#' @return Logical indicating SME support
#' @export
#' @family cpu_features
ggml_cpu_has_sme <- function() {
  .Call("R_ggml_cpu_has_sme", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - RISC-V Vector
#'
#' Check if the CPU supports RISC-V Vector extension.
#'
#' @return Logical indicating RISC-V V support
#' @export
#' @family cpu_features
ggml_cpu_has_riscv_v <- function() {
  .Call("R_ggml_cpu_has_riscv_v", PACKAGE = "ggmlR")
}

#' Get RISC-V Vector Length
#'
#' Returns the RISC-V RVV vector length in bytes (0 if not supported).
#'
#' @return Integer vector length in bytes
#' @export
#' @family cpu_features
ggml_cpu_get_rvv_vlen <- function() {
  .Call("R_ggml_cpu_get_rvv_vlen", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - VSX (PowerPC)
#'
#' Check if the CPU supports PowerPC VSX instructions.
#'
#' @return Logical indicating VSX support
#' @export
#' @family cpu_features
ggml_cpu_has_vsx <- function() {
  .Call("R_ggml_cpu_has_vsx", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - VXE (IBM z/Architecture)
#'
#' Check if the CPU supports IBM z/Architecture VXE instructions.
#'
#' @return Logical indicating VXE support
#' @export
#' @family cpu_features
ggml_cpu_has_vxe <- function() {
  .Call("R_ggml_cpu_has_vxe", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - WebAssembly SIMD
#'
#' Check if the CPU/environment supports WebAssembly SIMD.
#'
#' @return Logical indicating WASM SIMD support
#' @export
#' @family cpu_features
ggml_cpu_has_wasm_simd <- function() {
  .Call("R_ggml_cpu_has_wasm_simd", PACKAGE = "ggmlR")
}

#' CPU Feature Detection - Llamafile
#'
#' Check if llamafile optimizations are available.
#'
#' @return Logical indicating llamafile support
#' @export
#' @family cpu_features
ggml_cpu_has_llamafile <- function() {
  .Call("R_ggml_cpu_has_llamafile", PACKAGE = "ggmlR")
}

#' Get All CPU Features
#'
#' Returns a named list of all CPU feature detection results.
#' Useful for diagnostics and optimizing computation.
#'
#' @return Named list with feature names and logical values
#' @export
#' @family cpu_features
#' @examples
#' features <- ggml_cpu_features()
#' print(features)
#' # On typical x86-64: sse3=TRUE, avx=TRUE, avx2=TRUE, ...
ggml_cpu_features <- function() {
  list(
    # x86 features
    sse3 = ggml_cpu_has_sse3(),
    ssse3 = ggml_cpu_has_ssse3(),
    avx = ggml_cpu_has_avx(),
    avx_vnni = ggml_cpu_has_avx_vnni(),
    avx2 = ggml_cpu_has_avx2(),
    bmi2 = ggml_cpu_has_bmi2(),
    f16c = ggml_cpu_has_f16c(),
    fma = ggml_cpu_has_fma(),
    avx512 = ggml_cpu_has_avx512(),
    avx512_vbmi = ggml_cpu_has_avx512_vbmi(),
    avx512_vnni = ggml_cpu_has_avx512_vnni(),
    avx512_bf16 = ggml_cpu_has_avx512_bf16(),
    amx_int8 = ggml_cpu_has_amx_int8(),
    # ARM features
    neon = ggml_cpu_has_neon(),
    arm_fma = ggml_cpu_has_arm_fma(),
    fp16_va = ggml_cpu_has_fp16_va(),
    dotprod = ggml_cpu_has_dotprod(),
    matmul_int8 = ggml_cpu_has_matmul_int8(),
    sve = ggml_cpu_has_sve(),
    sve_cnt = ggml_cpu_get_sve_cnt(),
    sme = ggml_cpu_has_sme(),
    # Other
    riscv_v = ggml_cpu_has_riscv_v(),
    rvv_vlen = ggml_cpu_get_rvv_vlen(),
    vsx = ggml_cpu_has_vsx(),
    vxe = ggml_cpu_has_vxe(),
    wasm_simd = ggml_cpu_has_wasm_simd(),
    llamafile = ggml_cpu_has_llamafile()
  )
}

# ============================================================================
# Tensor Layout/Contiguity Functions
# ============================================================================

#' Check Tensor Contiguity (Dimension 0)
#'
#' Check if tensor is contiguous. Same as \code{ggml_is_contiguous}.
#'
#' @param tensor Tensor pointer
#' @return Logical indicating contiguity
#' @export
#' @family tensor_layout
ggml_is_contiguous_0 <- function(tensor) {
  .Call("R_ggml_is_contiguous_0", tensor, PACKAGE = "ggmlR")
}

#' Check Tensor Contiguity (Dimensions >= 1)
#'
#' Check if tensor is contiguous for dimensions >= 1.
#' Allows non-contiguous first dimension.
#'
#' @param tensor Tensor pointer
#' @return Logical indicating contiguity for dims >= 1
#' @export
#' @family tensor_layout
ggml_is_contiguous_1 <- function(tensor) {
  .Call("R_ggml_is_contiguous_1", tensor, PACKAGE = "ggmlR")
}

#' Check Tensor Contiguity (Dimensions >= 2)
#'
#' Check if tensor is contiguous for dimensions >= 2.
#' Allows non-contiguous first two dimensions.
#'
#' @param tensor Tensor pointer
#' @return Logical indicating contiguity for dims >= 2
#' @export
#' @family tensor_layout
ggml_is_contiguous_2 <- function(tensor) {
  .Call("R_ggml_is_contiguous_2", tensor, PACKAGE = "ggmlR")
}

#' Check If Tensor is Contiguously Allocated
#'
#' Check if tensor data is contiguously allocated in memory.
#' Different from contiguous layout - this checks the actual allocation.
#'
#' @param tensor Tensor pointer
#' @return Logical indicating if data is contiguously allocated
#' @export
#' @family tensor_layout
ggml_is_contiguously_allocated <- function(tensor) {
  .Call("R_ggml_is_contiguously_allocated", tensor, PACKAGE = "ggmlR")
}

#' Check Channel-wise Contiguity
#'
#' Check if tensor has contiguous channels (important for CNN operations).
#' Data for each channel should be stored contiguously.
#'
#' @param tensor Tensor pointer
#' @return Logical indicating channel-wise contiguity
#' @export
#' @family tensor_layout
ggml_is_contiguous_channels <- function(tensor) {
  .Call("R_ggml_is_contiguous_channels", tensor, PACKAGE = "ggmlR")
}

#' Check Row-wise Contiguity
#'
#' Check if tensor has contiguous rows (important for matrix operations).
#' Each row should be stored contiguously in memory.
#'
#' @param tensor Tensor pointer
#' @return Logical indicating row-wise contiguity
#' @export
#' @family tensor_layout
ggml_is_contiguous_rows <- function(tensor) {
  .Call("R_ggml_is_contiguous_rows", tensor, PACKAGE = "ggmlR")
}

#' Compare Tensor Strides
#'
#' Check if two tensors have the same stride pattern.
#' Useful for determining if tensors can share operations.
#'
#' @param a First tensor
#' @param b Second tensor
#' @return Logical indicating if strides are identical
#' @export
#' @family tensor_layout
ggml_are_same_stride <- function(a, b) {
  .Call("R_ggml_are_same_stride", a, b, PACKAGE = "ggmlR")
}

#' Check If Tensor Can Be Repeated
#'
#' Check if tensor \code{a} can be repeated (broadcast) to match tensor \code{b}.
#' Used for broadcasting operations.
#'
#' @param a Source tensor (smaller)
#' @param b Target tensor (larger or same size)
#' @return Logical indicating if a can be repeated to match b
#' @export
#' @family tensor_layout
#' @examples
#' \donttest{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
#' b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8)
#' ggml_can_repeat(a, b)  # TRUE - a can broadcast along dim 1
#' ggml_free(ctx)
#' }
ggml_can_repeat <- function(a, b) {
  .Call("R_ggml_can_repeat", a, b, PACKAGE = "ggmlR")
}

#' Count Equal Elements (Graph)
#'
#' Creates a graph node that counts equal elements between two tensors.
#' Useful for accuracy computation.
#'
#' @param ctx GGML context
#' @param a First tensor
#' @param b Second tensor (same shape as a)
#' @return Tensor containing the count of equal elements
#' @export
#' @family tensor_layout
#' @examples
#' \donttest{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' pred <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 100)
#' labels <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 100)
#' # ... set values ...
#' correct <- ggml_count_equal(ctx, pred, labels)
#' graph <- ggml_build_forward_expand(ctx, correct)
#' ggml_graph_compute(ctx, graph)
#' # correct now contains count of matching elements
#' ggml_free(ctx)
#' }
ggml_count_equal <- function(ctx, a, b) {
  .Call("R_ggml_count_equal", ctx, a, b, PACKAGE = "ggmlR")
}

# ============================================================================
# Timestep Embedding
# ============================================================================

#' Timestep Embedding (Graph Operation)
#'
#' Creates sinusoidal timestep embeddings as used in diffusion models.
#' Reference: CompVis/stable-diffusion util.py timestep_embedding
#'
#' @param ctx GGML context
#' @param timesteps Input tensor of timestep values [N]
#' @param dim Embedding dimension
#' @param max_period Maximum period for sinusoidal embedding (default 10000)
#' @return Tensor of shape [N, dim] with timestep embeddings
#' @export
ggml_timestep_embedding <- function(ctx, timesteps, dim, max_period = 10000L) {
  .Call("R_ggml_timestep_embedding", ctx, timesteps,
        as.integer(dim), as.integer(max_period), PACKAGE = "ggmlR")
}

# ============================================================================
# CPU-side Tensor Data Access (indexed)
# ============================================================================

#' Set Single Float Value by N-D Index
#'
#' Sets a single f32 value in the tensor at position [i0, i1, i2, i3].
#' This is a direct data write, not a graph operation.
#'
#' @param tensor Tensor pointer
#' @param i0,i1,i2,i3 Indices (0-based)
#' @param value Float value to set
#' @return NULL (invisible)
#' @export
ggml_set_f32_nd <- function(tensor, i0, i1 = 0, i2 = 0, i3 = 0, value) {
  invisible(.Call("R_ggml_set_f32_nd", tensor, as.numeric(i0), as.numeric(i1),
                  as.numeric(i2), as.numeric(i3), as.numeric(value), PACKAGE = "ggmlR"))
}

#' Get Single Float Value by N-D Index
#'
#' Gets a single f32 value from the tensor at position [i0, i1, i2, i3].
#' Works with any tensor type (auto-converts to float).
#'
#' @param tensor Tensor pointer
#' @param i0,i1,i2,i3 Indices (0-based)
#' @return Float value
#' @export
ggml_get_f32_nd <- function(tensor, i0, i1 = 0, i2 = 0, i3 = 0) {
  .Call("R_ggml_get_f32_nd", tensor, as.numeric(i0), as.numeric(i1),
        as.numeric(i2), as.numeric(i3), PACKAGE = "ggmlR")
}

#' Get Single Int32 Value by N-D Index
#'
#' Gets a single i32 value from the tensor at position [i0, i1, i2, i3].
#'
#' @param tensor Tensor pointer
#' @param i0,i1,i2,i3 Indices (0-based)
#' @return Integer value
#' @export
ggml_get_i32_nd <- function(tensor, i0, i1 = 0, i2 = 0, i3 = 0) {
  .Call("R_ggml_get_i32_nd", tensor, as.numeric(i0), as.numeric(i1),
        as.numeric(i2), as.numeric(i3), PACKAGE = "ggmlR")
}

#' Set Single Int32 Value by N-D Index
#'
#' Sets a single i32 value in the tensor at position [i0, i1, i2, i3].
#'
#' @param tensor Tensor pointer
#' @param i0,i1,i2,i3 Indices (0-based)
#' @param value Integer value to set
#' @return NULL (invisible)
#' @export
ggml_set_i32_nd <- function(tensor, i0, i1 = 0, i2 = 0, i3 = 0, value) {
  invisible(.Call("R_ggml_set_i32_nd", tensor, as.numeric(i0), as.numeric(i1),
                  as.numeric(i2), as.numeric(i3), as.integer(value), PACKAGE = "ggmlR"))
}

#' Get Tensor Strides (nb)
#'
#' Returns the byte strides for each dimension of the tensor.
#'
#' @param tensor Tensor pointer
#' @return Numeric vector of 4 stride values (nb0, nb1, nb2, nb3)
#' @export
ggml_tensor_nb <- function(tensor) {
  .Call("R_ggml_tensor_nb", tensor, PACKAGE = "ggmlR")
}

#' Backend Tensor Get and Sync
#'
#' Gets tensor data from a backend with synchronization.
#'
#' @param backend Backend pointer (or NULL for CPU)
#' @param tensor Tensor pointer
#' @param offset Byte offset (default 0)
#' @param size Number of bytes to read
#' @return Raw vector with tensor data
#' @export
ggml_backend_tensor_get_and_sync <- function(backend, tensor, offset = 0, size) {
  .Call("R_ggml_backend_tensor_get_and_sync", backend, tensor,
        as.numeric(offset), as.numeric(size), PACKAGE = "ggmlR")
}

#' Get First Float from Backend Tensor
#'
#' Reads the first f32 element from a backend tensor.
#'
#' @param tensor Tensor pointer
#' @return Float value
#' @export
ggml_backend_tensor_get_f32_first <- function(tensor) {
  .Call("R_ggml_backend_tensor_get_f32", tensor, PACKAGE = "ggmlR")
}

#' Count Tensors in Context
#'
#' Counts the number of tensors allocated in a context.
#'
#' @param ctx GGML context
#' @return Number of tensors
#' @export
ggml_tensor_num <- function(ctx) {
  .Call("R_ggml_tensor_num", ctx, PACKAGE = "ggmlR")
}

#' Copy Tensor Data
#'
#' Copies raw data from src tensor to dst tensor (must be same size).
#'
#' @param dst Destination tensor
#' @param src Source tensor
#' @return NULL (invisible)
#' @export
ggml_tensor_copy <- function(dst, src) {
  invisible(.Call("R_ggml_tensor_copy", dst, src, PACKAGE = "ggmlR"))
}

#' Fill Tensor with Scalar
#'
#' Sets all elements of a f32 tensor to a single value.
#'
#' @param tensor Tensor pointer
#' @param value Float value to fill with
#' @return NULL (invisible)
#' @export
ggml_tensor_set_f32_scalar <- function(tensor, value) {
  invisible(.Call("R_ggml_tensor_set_f32_scalar", tensor, as.numeric(value), PACKAGE = "ggmlR"))
}

#' Get First Tensor from Context
#'
#' @param ctx GGML context
#' @return Tensor pointer or NULL
#' @export
ggml_get_first_tensor <- function(ctx) {
  .Call("R_ggml_get_first_tensor", ctx, PACKAGE = "ggmlR")
}

#' Get Next Tensor from Context
#'
#' @param ctx GGML context
#' @param tensor Current tensor
#' @return Next tensor pointer or NULL
#' @export
ggml_get_next_tensor <- function(ctx, tensor) {
  .Call("R_ggml_get_next_tensor", ctx, tensor, PACKAGE = "ggmlR")
}
