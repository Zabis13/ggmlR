# ggml_layer_custom(): custom layers from an R function
#
# Turns an arbitrary composition of ggml operations into a reusable functional
# layer.  ggml_layer_custom() returns a *layer function* that pipes like any
# built-in ggml_layer_*(), and gradients flow through it automatically as long
# as every operation used inside `forward` implements a backward pass.
#
# Key points:
#
#   * `forward` takes (ctx, x) -- the compute context FIRST, then the parent
#     tensor.  Every ggml operation takes ctx as its first argument, so a
#     forward body reads ggml_mul(ctx, a, b), not a * b (ggml tensors do not
#     overload R's arithmetic operators).
#
#   * `output_shape = NULL` (default) means the layer preserves its input
#     shape -- correct for element-wise activations.  Pass an integer vector,
#     or a function of the input shape, when `forward` changes the shape.
#
#   * Custom layers are stateless: they own no trainable parameters.  Put
#     learnable weights in a neighbouring ggml_layer_dense(), or use the
#     autograd API (ag_*) when a layer needs its own.
#
# Architecture:
#
#   input(8) -> Dense(16) -> mish() -> Dense(16) -> scaled_swish() ->
#               Dense(2, softmax)
#
# Task: binary classification of whether a vector's sum exceeds a threshold.

library(ggmlR)

invisible(ggml_set_n_threads(1L))

# ---------------------------------------------------------------------------
# 1. Define two custom layers
# ---------------------------------------------------------------------------

# Mish: x * tanh(softplus(x)).  Every op has a backward, so the layer trains.
layer_mish <- ggml_layer_custom(
  name    = "mish",
  forward = function(ctx, x) ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)))
)

# Swish/SiLU with a fixed gain: 1.5 * (x * sigmoid(x)).  Shows that a custom
# forward can close over ordinary R values -- `gain` is baked into the graph
# as a constant, it is NOT a trainable parameter.
make_scaled_swish <- function(gain) {
  ggml_layer_custom(
    name    = "scaled_swish",
    forward = function(ctx, x) {
      ggml_scale(ctx, ggml_mul(ctx, x, ggml_sigmoid(ctx, x)), gain)
    }
  )
}
layer_scaled_swish <- make_scaled_swish(1.5)

# ---------------------------------------------------------------------------
# 2. Synthetic dataset
# ---------------------------------------------------------------------------

set.seed(42)

N      <- 512L   # divisible by batch_size = 32
N_FEAT <- 8L

x <- matrix(runif(N * N_FEAT), nrow = N, ncol = N_FEAT)
y <- matrix(0.0, nrow = N, ncol = 2L)
for (i in seq_len(N)) {
  y[i, if (sum(x[i, ]) > N_FEAT / 2) 1L else 2L] <- 1.0
}

cat(sprintf("Dataset: %d samples, %d features, class balance %.2f / %.2f\n\n",
            N, N_FEAT, mean(y[, 1]), mean(y[, 2])))

# ---------------------------------------------------------------------------
# 3. Build the model -- custom layers pipe like any other layer
# ---------------------------------------------------------------------------

inp <- ggml_input(shape = N_FEAT, name = "features")

out <- inp |>
  ggml_layer_dense(16L, activation = "relu") |>
  layer_mish() |>
  ggml_layer_dense(16L) |>
  layer_scaled_swish() |>
  ggml_layer_dense(2L, activation = "softmax")

model <- ggml_model(inputs = inp, outputs = out)
print(model)

# ---------------------------------------------------------------------------
# 4. Compile and train
# ---------------------------------------------------------------------------

# NOTE: weights are drawn at ggml_compile() time, so set.seed() must come
# BEFORE the compile call to make a run reproducible.
set.seed(1L)
model <- ggml_compile(model,
                      optimizer = "adam",
                      loss      = "categorical_crossentropy",
                      metrics   = "accuracy")

model <- ggml_fit(model, x, y, batch_size = 32L, epochs = 10L, verbose = FALSE)

loss <- model$history$train_loss
acc  <- model$history$train_accuracy

cat("Training history:\n")
for (ep in seq_along(loss)) {
  cat(sprintf("  epoch %2d/%d   loss = %.4f   acc = %.3f\n",
              ep, length(loss), loss[ep], acc[ep]))
}
cat(sprintf("\nloss: %.4f -> %.4f   accuracy: %.3f -> %.3f\n",
            loss[1], loss[length(loss)], acc[1], acc[length(acc)]))

if (loss[length(loss)] < loss[1]) {
  cat("\nOK: gradients flow through both custom layers.\n")
} else {
  cat("\nFAIL: loss did not decrease -- check the custom forward bodies.\n")
}

# ---------------------------------------------------------------------------
# 5. Predict
# ---------------------------------------------------------------------------

# predict() runs on whole batches, so feed at least batch_size rows.
pred <- ggml_predict(model, x[1:32, , drop = FALSE])
cat("\nFirst 5 predictions (softmax) vs. true labels:\n")
print(data.frame(
  p_class1 = round(pred[1:5, 1], 4),
  p_class2 = round(pred[1:5, 2], 4),
  true     = max.col(y[1:5, , drop = FALSE])
))

# ---------------------------------------------------------------------------
# 6. A shape-changing custom layer
# ---------------------------------------------------------------------------

# When `forward` changes the shape, declare it via output_shape -- otherwise
# the graph builder keeps assuming the input shape and later layers break.
# Here: sum-of-squares pooling over pairs of adjacent features, 8 -> 4.
layer_pair_energy <- ggml_layer_custom(
  name         = "pair_energy",
  output_shape = 4L,
  forward      = function(ctx, x) {
    sq <- ggml_sqr(ctx, x)                       # (8, batch)
    # ggml pools only by max or avg: the sum of a pair is twice its mean
    ggml_scale(ctx, ggml_pool_1d(ctx, sq, GGML_OP_POOL_AVG, k0 = 2L, s0 = 2L), 2)
  }
)

cat("\nShape-changing custom layer (8 -> 4):\n")
inp2 <- ggml_input(shape = N_FEAT)
out2 <- inp2 |>
  layer_pair_energy() |>
  ggml_layer_dense(2L, activation = "softmax")
m2 <- ggml_model(inputs = inp2, outputs = out2)
print(m2)
