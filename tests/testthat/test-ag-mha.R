# Tests for ag_multihead_attention

# ============================================================================
# Construction
# ============================================================================

test_that("ag_multihead_attention constructs without error", {
  mha <- ag_multihead_attention(64L, 8L)
  expect_s3_class(mha, "ag_multihead_attention")
  expect_equal(mha$d_model, 64L)
  expect_equal(mha$n_heads, 8L)
  expect_equal(mha$d_k, 8L)
})

test_that("ag_multihead_attention rejects d_model not divisible by n_heads", {
  expect_error(ag_multihead_attention(65L, 8L))
})

test_that("parameters() returns W_q, W_k, W_v, W_o, b_o", {
  mha <- ag_multihead_attention(16L, 2L)
  p   <- mha$parameters()
  expect_true(all(c("W_q", "W_k", "W_v", "W_o", "b_o") %in% names(p)))
  expect_equal(length(p), 5L)
})

test_that("parameters() without bias returns 4 params", {
  mha <- ag_multihead_attention(16L, 2L, bias = FALSE)
  p   <- mha$parameters()
  expect_equal(length(p), 4L)
  expect_false("b_o" %in% names(p))
})

# ============================================================================
# Forward pass shape
# ============================================================================

test_that("self-attention output shape matches input", {
  set.seed(1)
  mha <- ag_multihead_attention(32L, 4L)
  x   <- ag_tensor(matrix(rnorm(32 * 7), 32, 7))
  out <- mha$forward(x)
  expect_equal(dim(ggmlR:::.ag_data(out)), c(32L, 7L))
})

test_that("cross-attention output shape is [d_model, seq_q]", {
  set.seed(2)
  mha <- ag_multihead_attention(32L, 4L)
  q   <- ag_tensor(matrix(rnorm(32 * 5), 32, 5))
  kv  <- ag_tensor(matrix(rnorm(32 * 9), 32, 9))
  out <- mha$forward(q, kv, kv)
  expect_equal(dim(ggmlR:::.ag_data(out)), c(32L, 5L))
})

test_that("single-head attention works (n_heads = 1)", {
  mha <- ag_multihead_attention(16L, 1L)
  x   <- ag_tensor(matrix(rnorm(16 * 4), 16, 4))
  out <- mha$forward(x)
  expect_equal(dim(ggmlR:::.ag_data(out)), c(16L, 4L))
})

# ============================================================================
# Causal mask
# ============================================================================

test_that("causal mask: future tokens receive near-zero attention weight", {
  set.seed(3)
  # Use 1 head for interpretability
  mha <- ag_multihead_attention(8L, 1L, bias = FALSE)
  seq_len <- 5L

  # Set up so that attention weights are directly inspectable:
  # Override W_q, W_k to identity-like (result is interpretable)
  # Just check that outputs differ with vs without mask
  x      <- ag_tensor(matrix(rnorm(8 * seq_len), 8, seq_len))
  out_no_mask <- mha$forward(x, causal_mask = FALSE)
  out_masked  <- mha$forward(x, causal_mask = TRUE)

  d_no <- ggmlR:::.ag_data(out_no_mask)
  d_mk <- ggmlR:::.ag_data(out_masked)

  # Outputs must differ (mask changes attention distribution)
  expect_false(isTRUE(all.equal(d_no, d_mk, tolerance = 1e-6)))
  # Both outputs have correct shape
  expect_equal(dim(d_mk), c(8L, seq_len))
})

# ============================================================================
# Backward / gradients
# ============================================================================

test_that("backward passes through MHA without error", {
  set.seed(4)
  mha  <- ag_multihead_attention(16L, 2L)
  x    <- ag_tensor(matrix(rnorm(16 * 4), 16, 4))
  y    <- matrix(0.0, 16, 4)

  with_grad_tape({
    out  <- mha$forward(x)
    loss <- ag_mse_loss(out, y)
  })
  grads <- backward(loss)

  # All parameter gradients must be non-NULL and non-zero
  for (nm in c("W_q", "W_k", "W_v", "W_o")) {
    p   <- mha$parameters()[[nm]]
    key <- as.character(p$id)
    g   <- get0(key, envir = grads)
    expect_false(is.null(g), info = paste("gradient missing for", nm))
    expect_true(any(g != 0), info = paste("gradient zero for", nm))
  }
})

test_that("optimizer step reduces MHA loss over 10 iterations", {
  set.seed(5)
  d_model <- 16L; n_heads <- 2L; seq_len <- 6L
  mha <- ag_multihead_attention(d_model, n_heads)
  opt <- optimizer_adam(mha$parameters(), lr = 1e-3)

  x_mat <- matrix(rnorm(d_model * seq_len), d_model, seq_len)
  y_mat <- matrix(rnorm(d_model * seq_len), d_model, seq_len)

  losses <- numeric(10L)
  for (i in seq_len(10L)) {
    x <- ag_tensor(x_mat)
    with_grad_tape({
      out  <- mha$forward(x)
      loss <- ag_mse_loss(out, y_mat)
    })
    grads <- backward(loss)
    opt$step(grads)
    opt$zero_grad()
    losses[i] <- as.numeric(ggmlR:::.ag_data(loss))
  }

  expect_lt(losses[10L], losses[1L])
})

# Finite differences against the REAL mha$forward().
#
# The previous version of this check rebuilt an equivalent graph by hand from
# ag_matmul/ag_softmax and differentiated that. It therefore tested the base
# ops -- already covered in test-autograd-ops.R -- and not the layer: the
# hand-written replica hard-coded the scale, skipped b_o, skipped the causal
# mask and used its own row slicing, so any of those could be wrong in the
# layer and the check would still pass. ag_batch_norm shipped a wrong gradient
# behind exactly that kind of gap. These call the layer itself.

# Swap the layer's parameters for the tensors gradcheck perturbs, run the real
# forward, then put the originals back.
mha_loss_fn <- function(mha, q, k = NULL, v = NULL, causal = FALSE, w = NULL) {
  function(ins) {
    orig <- list(W_q = mha$W_q, W_k = mha$W_k, W_v = mha$W_v,
                 W_o = mha$W_o, b_o = mha$b_o)
    on.exit({
      mha$W_q <- orig$W_q; mha$W_k <- orig$W_k; mha$W_v <- orig$W_v
      mha$W_o <- orig$W_o; mha$b_o <- orig$b_o
    }, add = TRUE)

    for (nm in names(ins)) {
      if (nm %in% c("W_q", "W_k", "W_v", "W_o", "b_o")) mha[[nm]] <- ins[[nm]]
    }
    qq <- if (!is.null(ins$q)) ins$q else q
    out <- if (is.null(k)) mha$forward(qq, causal_mask = causal)
           else            mha$forward(qq, k, v, causal_mask = causal)
    # A non-constant weight matrix: a plain sum lets sign-symmetric errors
    # cancel, which is how a wrong gradient survives a gradcheck.
    ag_sum(ag_mul(out, ag_tensor(w)))
  }
}

test_that("gradcheck passes for the real MHA forward (weights, self-attention)", {
  set.seed(6)
  mha <- ag_multihead_attention(4L, 2L, bias = FALSE)
  x   <- ag_tensor(matrix(rnorm(4 * 3), 4, 3))
  w   <- matrix(seq(0.2, by = 0.09, length.out = 12), 4, 3)

  p  <- mha$parameters()
  ok <- ag_gradcheck(
    fn     = mha_loss_fn(mha, x, w = w),
    inputs = list(W_q = p$W_q, W_k = p$W_k, W_v = p$W_v, W_o = p$W_o),
    atol   = 1e-3, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck passes for the MHA input, not just its weights", {
  # The old check never perturbed x, so the gradient flowing back to the input
  # -- through the slicing, the softmax and the concat -- was unverified.
  set.seed(7)
  mha <- ag_multihead_attention(4L, 2L, bias = FALSE)
  x   <- ag_param(matrix(rnorm(4 * 3), 4, 3))
  w   <- matrix(seq(0.15, by = 0.11, length.out = 12), 4, 3)

  ok <- ag_gradcheck(
    fn     = mha_loss_fn(mha, NULL, w = w),
    inputs = list(q = x),
    atol   = 1e-3, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck covers the output bias b_o", {
  # bias = TRUE is the constructor default, yet the old gradcheck used
  # bias = FALSE and left the ag_add(out, b_o) branch unchecked.
  set.seed(8)
  mha <- ag_multihead_attention(4L, 2L, bias = TRUE)
  x   <- ag_tensor(matrix(rnorm(4 * 3), 4, 3))
  w   <- matrix(seq(0.25, by = 0.07, length.out = 12), 4, 3)

  p  <- mha$parameters()
  ok <- ag_gradcheck(
    fn     = mha_loss_fn(mha, x, w = w),
    inputs = list(W_o = p$W_o, b_o = p$b_o),
    atol   = 1e-3, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck passes with the causal mask on", {
  # -Inf enters the softmax here; the gradient must still match finite
  # differences, not merely be finite (that is test-ag-mha.R's other check).
  set.seed(9)
  mha <- ag_multihead_attention(4L, 2L, bias = FALSE)
  x   <- ag_tensor(matrix(rnorm(4 * 4), 4, 4))
  w   <- matrix(seq(0.3, by = 0.05, length.out = 16), 4, 4)

  p  <- mha$parameters()
  ok <- ag_gradcheck(
    fn     = mha_loss_fn(mha, x, causal = TRUE, w = w),
    inputs = list(W_q = p$W_q, W_v = p$W_v),
    atol   = 1e-3, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck passes for cross-attention (q, k, v distinct)", {
  # k and v are separate tensors here, so W_k and W_v see a different input
  # than W_q -- a case self-attention cannot distinguish.
  set.seed(10)
  mha <- ag_multihead_attention(4L, 2L, bias = FALSE)
  q   <- ag_tensor(matrix(rnorm(4 * 3), 4, 3))
  kv  <- ag_tensor(matrix(rnorm(4 * 5), 4, 5))
  w   <- matrix(seq(0.18, by = 0.06, length.out = 12), 4, 3)

  p  <- mha$parameters()
  ok <- ag_gradcheck(
    fn     = mha_loss_fn(mha, q, kv, kv, w = w),
    inputs = list(W_q = p$W_q, W_k = p$W_k, W_v = p$W_v),
    atol   = 1e-3, quiet = TRUE
  )
  expect_true(ok)
})

test_that("gradcheck passes for 4 heads (scale depends on d_k)", {
  # env$scale = 1/sqrt(d_k). The old replica hard-coded 1/sqrt(2), which is
  # only right for d_k = 2, so a wrong scale in the layer went unnoticed.
  set.seed(11)
  mha <- ag_multihead_attention(8L, 4L, bias = FALSE)
  x   <- ag_tensor(matrix(rnorm(8 * 3), 8, 3))
  w   <- matrix(seq(0.1, by = 0.03, length.out = 24), 8, 3)

  p  <- mha$parameters()
  ok <- ag_gradcheck(
    fn     = mha_loss_fn(mha, x, w = w),
    inputs = list(W_q = p$W_q, W_o = p$W_o),
    atol   = 1e-3, quiet = TRUE
  )
  expect_true(ok)
})
# ============================================================================
# train / eval mode
# ============================================================================

test_that("ag_train / ag_eval toggle training flag", {
  mha <- ag_multihead_attention(16L, 2L)
  expect_true(mha$training)
  ag_eval(mha)
  expect_false(mha$training)
  ag_train(mha)
  expect_true(mha$training)
})

test_that("dropout=0.5 changes output in training vs eval", {
  set.seed(7)
  mha <- ag_multihead_attention(16L, 2L, dropout = 0.5)
  x   <- ag_tensor(matrix(rnorm(16 * 6), 16, 6))

  ag_train(mha)
  out_train <- ggmlR:::.ag_data(mha$forward(x))
  ag_eval(mha)
  out_eval  <- ggmlR:::.ag_data(mha$forward(x))

  # Training output (stochastic) differs from eval (deterministic)
  expect_false(isTRUE(all.equal(out_train, out_eval, tolerance = 1e-6)))
})

# ============================================================================
# ag_sequential integration
# ============================================================================

test_that("ag_multihead_attention works inside ag_sequential", {
  set.seed(8)
  model <- ag_sequential(
    ag_multihead_attention(16L, 2L),
    ag_dropout(0.0)
  )
  x   <- ag_tensor(matrix(rnorm(16 * 5), 16, 5))
  out <- model$forward(x)
  expect_equal(dim(ggmlR:::.ag_data(out)), c(16L, 5L))

  p <- model$parameters()
  expect_true(length(p) >= 5L)
})

# ============================================================================
# Causal mask: -Inf must not reach the gradients
# ============================================================================

test_that("causal mask: backward produces finite gradients (seq_q == seq_kv)", {
  # The mask injects -Inf before the softmax. Partial masking is safe -- p is
  # exactly 0 there, so the softmax backward contributes 0 * (...) -- but that
  # holds only while no row is fully masked. This pins the safe case down.
  set.seed(40)
  mha <- ag_multihead_attention(8L, 2L, bias = FALSE)
  x   <- ag_param(matrix(rnorm(8 * 5), 8, 5))

  with_grad_tape({
    out  <- mha$forward(x, causal_mask = TRUE)
    loss <- ag_mse_loss(out, matrix(0.0, 8, 5))
  })
  expect_true(is.finite(as.numeric(ggmlR:::.ag_data(loss))))

  grads <- backward(loss)

  g_x <- get0(as.character(x$id), envir = grads)
  expect_false(is.null(g_x))
  expect_true(all(is.finite(g_x)))

  for (nm in c("W_q", "W_k", "W_v", "W_o")) {
    g <- get0(as.character(mha$parameters()[[nm]]$id), envir = grads)
    expect_false(is.null(g), info = paste("gradient missing for", nm))
    expect_true(all(is.finite(g)), info = paste("non-finite gradient for", nm))
  }
})

test_that("causal mask: backward is finite for cross-attention shapes", {
  # seq_kv > seq_q and seq_q > seq_kv both keep the diagonal open, so neither
  # can produce a fully masked row. Checked here because the mask is built from
  # two independent lengths and only the square case is exercised elsewhere.
  for (dims in list(c(3L, 6L), c(6L, 3L), c(1L, 4L), c(4L, 1L))) {
    seq_q <- dims[[1L]]; seq_kv <- dims[[2L]]
    set.seed(41)
    mha <- ag_multihead_attention(4L, 2L, bias = FALSE)
    q   <- ag_param(matrix(rnorm(4 * seq_q),  4, seq_q))
    kv  <- ag_tensor(matrix(rnorm(4 * seq_kv), 4, seq_kv))

    with_grad_tape({
      out  <- mha$forward(q, kv, kv, causal_mask = TRUE)
      loss <- ag_mse_loss(out, matrix(0.0, 4, seq_q))
    })
    info <- sprintf("seq_q=%d seq_kv=%d", seq_q, seq_kv)
    expect_true(is.finite(as.numeric(ggmlR:::.ag_data(loss))), info = info)

    g_q <- get0(as.character(q$id), envir = backward(loss))
    expect_false(is.null(g_q), info = info)
    expect_true(all(is.finite(g_q)), info = info)
  }
})

test_that("causal mask: masked positions get exactly zero attention weight", {
  # Not just "finite" but "actually masking": the first query may attend only
  # to the first key, so with one head its output must equal V's first column.
  set.seed(42)
  mha <- ag_multihead_attention(4L, 1L, bias = FALSE)
  x   <- ag_tensor(matrix(rnorm(4 * 4), 4, 4))

  out <- ggmlR:::.ag_data(mha$forward(x, causal_mask = TRUE))

  W_v <- ggmlR:::.ag_data(mha$parameters()$W_v)
  W_o <- ggmlR:::.ag_data(mha$parameters()$W_o)
  v1  <- (W_v %*% ggmlR:::.ag_data(x))[, 1L, drop = FALSE]

  expect_equal(out[, 1L], as.numeric(W_o %*% v1), tolerance = 1e-6)
})
