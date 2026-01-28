#' Simple Transformer Layer Example for LLM
#' This demonstrates how to build a basic transformer-like computation

library(ggmlR)

cat("=== Simple Transformer Layer Example ===\n\n")

# Parameters
batch_size <- 1
seq_len <- 8
d_model <- 64
d_ff <- 256

# Initialize context (allocate enough memory)
mem_size <- 256 * 1024 * 1024  # 256MB
ctx <- ggml_init(mem_size)

cat("Context initialized with", mem_size / (1024*1024), "MB\n\n")

# ============================================================================
# 1. Input embeddings
# ============================================================================
cat("Creating input tensor...\n")
input <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, seq_len)
input_data <- rnorm(d_model * seq_len, mean = 0, sd = 0.02)
ggml_set_f32(input, input_data)

# ============================================================================
# 2. Layer Normalization
# ============================================================================
cat("Applying layer normalization...\n")
norm1 <- ggml_rms_norm(ctx, input, eps = 1e-5)

# ============================================================================
# 3. Feed-Forward Network
# ============================================================================
cat("Creating FFN weights...\n")

# First linear layer: d_model -> d_ff
W1 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, d_ff)
ggml_set_f32(W1, rnorm(d_model * d_ff, sd = sqrt(2.0 / d_model)))

# Second linear layer: d_ff -> d_model
W2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_ff, d_model)
ggml_set_f32(W2, rnorm(d_ff * d_model, sd = sqrt(2.0 / d_ff)))

cat("Forward pass through FFN...\n")

# FFN: W2 * GELU(W1 * x)
h1 <- ggml_mul_mat(ctx, W1, norm1)
h1_act <- ggml_gelu(ctx, h1)
h2 <- ggml_mul_mat(ctx, W2, h1_act)

# ============================================================================
# 4. Residual connection
# ============================================================================
cat("Adding residual connection...\n")
output <- ggml_add(ctx, input, h2)

# Final layer norm
output_norm <- ggml_rms_norm(ctx, output, eps = 1e-5)

# ============================================================================
# 5. Compute graph
# ============================================================================
cat("Building computation graph...\n")
graph <- ggml_build_forward_expand(ctx, output_norm)

cat("Computing graph...\n")
system.time({
  ggml_graph_compute(ctx, graph)
})

# ============================================================================
# 6. Get results
# ============================================================================
cat("\nGetting results...\n")
result <- ggml_get_f32(output_norm)

cat("Output shape: [", d_model, "x", seq_len, "]\n")
cat("Output stats:\n")
cat("  Mean:", mean(result), "\n")
cat("  Std:", sd(result), "\n")
cat("  Min:", min(result), "\n")
cat("  Max:", max(result), "\n")

cat("\nFirst 10 values:\n")
print(head(result, 10))

# Cleanup
ggml_free(ctx)
cat("\n=== Done ===\n")
