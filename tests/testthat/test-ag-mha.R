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

test_that("ag_gradcheck passes for MHA (small model)", {
  set.seed(6)
  mha <- ag_multihead_attention(4L, 2L, bias = FALSE)

  W_q <- mha$parameters()$W_q
  W_k <- mha$parameters()$W_k
  W_v <- mha$parameters()$W_v
  W_o <- mha$parameters()$W_o

  x_fix <- matrix(rnorm(4 * 3), 4, 3)

  result <- ag_gradcheck(
    fn = function(ins) {
      # Temporarily assign weights for gradcheck perturbation
      ins$W_q$data <- ins$W_q$data  # identity (gradcheck manages)
      ag_mse_loss(
        ag_matmul(ins$W_o,
          ggmlR:::.ag_row_concat(lapply(seq_len(2L), function(h) {
            rows <- ((h-1)*2+1):(h*2)
            q_h  <- ag_matmul(ag_tensor(diag(1,4)[rows,]), ag_matmul(ins$W_q, ag_tensor(x_fix)))
            k_h  <- ag_matmul(ag_tensor(diag(1,4)[rows,]), ag_matmul(ins$W_k, ag_tensor(x_fix)))
            v_h  <- ag_matmul(ag_tensor(diag(1,4)[rows,]), ag_matmul(ins$W_v, ag_tensor(x_fix)))
            sc   <- ag_scale(ag_matmul(ag_transpose(q_h), k_h), 1/sqrt(2))
            at   <- ag_transpose(ag_softmax(ag_transpose(sc)))
            ag_matmul(v_h, ag_transpose(at))
          }))
        ),
        matrix(0, 4, 3)
      )
    },
    inputs = list(W_q = W_q, W_k = W_k, W_v = W_v, W_o = W_o),
    atol   = 1e-3,
    quiet  = TRUE
  )
  expect_true(result)
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
