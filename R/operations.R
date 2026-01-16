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
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(5, 4, 3, 2, 1))
#' result <- ggml_cpu_add(a, b)
#' print(result)  # [6, 6, 6, 6, 6]
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
#' \dontrun{
#' ctx <- ggml_init(1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(2, 2, 2, 2, 2))
#' result <- ggml_cpu_mul(a, b)
#' print(result)  # [2, 4, 6, 8, 10]
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' b <- ggml_dup(ctx, a)  # Create copy operation
#' graph <- ggml_build_forward_expand(ctx, b)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(b)  # Should be c(1, 2, 3, 4, 5)
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(b, c(5, 4, 3, 2, 1))
#' c <- ggml_add(ctx, a, b)
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(c)
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
#' scalar <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
#' ggml_set_f32(a, c(1, 2, 3, 4, 5))
#' ggml_set_f32(scalar, 10)
#' c <- ggml_add1(ctx, a, scalar)  # Add 10 to all elements
#' graph <- ggml_build_forward_expand(ctx, c)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(c)  # c(11, 12, 13, 14, 15)
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#' # Create 10x20 and 20x5 matrices
#' A <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 20)
#' B <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 20, 5)
#' ggml_set_f32(A, rnorm(200))
#' ggml_set_f32(B, rnorm(100))
#' # Result will be 10x5
#' C <- ggml_mul_mat(ctx, A, B)
#' graph <- ggml_build_forward_expand(ctx, C)
#' ggml_graph_compute(ctx, graph)
#' result <- ggml_get_f32(C)
#' ggml_free(ctx)
#' }
ggml_mul_mat <- function(ctx, a, b) {
  .Call("R_ggml_mul_mat", ctx, a, b, PACKAGE = "ggmlR")
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
ggml_tanh <- function(ctx, a) {
  .Call("R_ggml_tanh", ctx, a, PACKAGE = "ggmlR")
}

# ============================================================================
# Normalization Functions
# ============================================================================

#' Layer Normalization (Graph)
#' 
#' Creates a graph node for layer normalization.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param eps Epsilon value for numerical stability (default: 1e-5)
#' @return Tensor representing the layer normalization operation
#' @export
ggml_norm <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_norm", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
}

#' RMS Normalization (Graph)
#'
#' Creates a graph node for RMS (Root Mean Square) normalization.
#' CRITICAL for LLaMA models.
#'
#' @param ctx GGML context
#' @param a Input tensor
#' @param eps Epsilon value for numerical stability (default: 1e-5)
#' @return Tensor representing the RMS normalization operation
#' @export
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
#' @export
ggml_rms_norm_inplace <- function(ctx, a, eps = 1e-5) {
  .Call("R_ggml_rms_norm_inplace", ctx, a, as.numeric(eps), PACKAGE = "ggmlR")
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
#' \dontrun{
#' ctx <- ggml_init(64 * 1024 * 1024)
#'
#' # Attention scores: [head_dim, seq_len, n_heads, batch]
#' scores <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 128, 8, 1)
#'
#' # Causal mask (optional)
#' mask <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128)
#'
#' # Apply softmax with scaling
#' scale <- 1.0 / sqrt(64)  # 1/sqrt(head_dim)
#' attn <- ggml_soft_max_ext(ctx, scores, mask, scale, max_bias = 0.0)
#'
#' ggml_free(ctx)
#' }
#' @export
ggml_soft_max_ext <- function(ctx, a, mask = NULL, scale = 1.0, max_bias = 0.0) {
  .Call("R_ggml_soft_max_ext", ctx, a, mask,
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' @details
#' \itemize{
#'   \item \code{GGML_GLU_OP_REGLU}: ReGLU - ReLU gating
#'   \item \code{GGML_GLU_OP_GEGLU}: GeGLU - GELU gating (used in GPT-NeoX, etc.)
#'   \item \code{GGML_GLU_OP_SWIGLU}: SwiGLU - SiLU/Swish gating (used in LLaMA)
#'   \item \code{GGML_GLU_OP_GEGLU_QUICK}: GeGLU with fast approximation
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#'
#' # Create embedding matrix: 10 tokens, 4-dim embeddings
#' embeddings <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10)
#' ggml_set_f32(embeddings, rnorm(40))
#'
#' # Token indices to look up
#' indices <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3)
#' ggml_set_i32(indices, c(0, 5, 2))
#'
#' # Get embeddings for tokens 0, 5, 2
#' result <- ggml_get_rows(ctx, embeddings, indices)
#' # result shape: [4, 3]
#'
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#'
#' # Create attention scores matrix
#' scores <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4)
#' ggml_set_f32(scores, rep(1, 16))
#'
#' # Apply causal mask
#' masked <- ggml_diag_mask_inf(ctx, scores, 0)
#' # After computation, upper triangle will be -Inf
#'
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
#' @description
#' Constants for RoPE (Rotary Position Embedding) modes.
#'
#' @details
#' - GGML_ROPE_TYPE_NORM (0): Standard RoPE as in original paper
#' - GGML_ROPE_TYPE_NEOX (2): GPT-NeoX style RoPE (interleaved differently)
#' - GGML_ROPE_TYPE_MROPE (8): Multi-RoPE for models like Qwen2-VL
#' - GGML_ROPE_TYPE_VISION (24): Vision model RoPE
#'
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#'
#' # Query tensor: head_dim=8, n_head=4, seq_len=16, batch=1
#' q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 4, 16, 1)
#' ggml_set_f32(q, rnorm(8 * 4 * 16))
#'
#' # Position indices
#' pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16)
#' ggml_set_i32(pos, 0:15)
#'
#' # Apply RoPE
#' q_rope <- ggml_rope(ctx, q, pos, 8, GGML_ROPE_TYPE_NORM)
#'
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
#' \dontrun{
#' ctx <- ggml_init(16 * 1024 * 1024)
#'
#' q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 8, 32, 1)
#' pos <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 32)
#' ggml_set_i32(pos, 0:31)
#'
#' # Standard RoPE with default freq_base
#' q_rope <- ggml_rope_ext(ctx, q, pos, NULL,
#'                         n_dims = 64, mode = 0L,
#'                         n_ctx_orig = 4096,
#'                         freq_base = 10000, freq_scale = 1.0,
#'                         ext_factor = 0.0, attn_factor = 1.0,
#'                         beta_fast = 32, beta_slow = 1)
#'
#' ggml_free(ctx)
#' }
#' @export
ggml_rope_ext <- function(ctx, a, b, c = NULL,
                          n_dims, mode = 0L, n_ctx_orig,
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
#' \dontrun{
#' ctx <- ggml_init(64 * 1024 * 1024)
#'
#' head_dim <- 64
#' n_head <- 8
#' n_head_kv <- 2  # GQA with 4:1 ratio
#' seq_len <- 32
#'
#' q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, seq_len, 1)
#' k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)
#' v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, seq_len, 1)
#'
#' ggml_set_f32(q, rnorm(head_dim * n_head * seq_len))
#' ggml_set_f32(k, rnorm(head_dim * n_head_kv * seq_len))
#' ggml_set_f32(v, rnorm(head_dim * n_head_kv * seq_len))
#'
#' # Scale = 1/sqrt(head_dim)
#' scale <- 1.0 / sqrt(head_dim)
#'
#' # Compute attention
#' out <- ggml_flash_attn_ext(ctx, q, k, v, NULL, scale, 0.0, 0.0)
#'
#' ggml_free(ctx)
#' }
#' @export
ggml_flash_attn_ext <- function(ctx, q, k, v, mask = NULL,
                                scale, max_bias = 0.0, logit_softcap = 0.0) {
  .Call("R_ggml_flash_attn_ext", ctx, q, k, v, mask,
        as.numeric(scale), as.numeric(max_bias), as.numeric(logit_softcap),
        PACKAGE = "ggmlR")
}
