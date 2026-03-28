# Chain tests: Popular ML patterns
# 1. Autograd regression (MSE loss)
# 2. Transfer learning: freeze backbone, train head only
# 3. Sequential save/load with BatchNorm (running stats preserved)
# 4. Functional multi-input classification (two feature groups → merge)

# ── 1. Autograd regression: MSE loss on synthetic data ─────

test_that("chain pattern: autograd regression with MSE loss", {
  # y = 2*x1 + 3*x2 + 1 + noise
  set.seed(42)
  n <- 80L
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y_true <- 2 * x1 + 3 * x2 + 1 + rnorm(n, sd = 0.1)
  x_cm <- rbind(x1, x2)                    # [2, n]
  y_cm <- matrix(y_true, nrow = 1)          # [1, n]

  m <- ag_sequential(
    ag_linear(2L, 16L, activation = "relu"),
    ag_linear(16L, 1L)
  )
  opt <- optimizer_adam(m$parameters(), lr = 1e-2)
  BS <- 16L

  losses <- numeric(50)
  ag_train(m)
  for (ep in seq_len(50L)) {
    perm <- sample(n)
    ep_loss <- 0; nb <- 0L
    for (b in seq_len(ceiling(n / BS))) {
      idx <- perm[((b-1L)*BS+1L):min(b*BS, n)]
      xb <- ag_tensor(x_cm[, idx, drop = FALSE])
      yb <- y_cm[, idx, drop = FALSE]
      with_grad_tape({
        pred <- m$forward(xb)
        loss <- ag_mse_loss(pred, yb)
      })
      grads <- backward(loss)
      opt$step(grads)
      opt$zero_grad()
      ep_loss <- ep_loss + loss$data[1]
      nb <- nb + 1L
    }
    losses[ep] <- ep_loss / nb
  }

  # Loss should decrease significantly
  expect_true(mean(tail(losses, 5)) < mean(head(losses, 5)))

  # Predictions should be reasonable
  ag_eval(m)
  pred <- m$forward(ag_tensor(x_cm))$data
  rmse <- sqrt(mean((pred[1,] - y_true)^2))
  expect_true(rmse < 1.0)  # should be much better than naive
})

# ── 2. Transfer learning: freeze backbone, train head ──────

test_that("chain pattern: transfer learning — freeze backbone, train head", {
  set.seed(42)
  n <- 80L
  x_all <- rbind(matrix(rnorm(n, -2, 0.5), n/2, 2),
                 matrix(rnorm(n,  2, 0.5), n/2, 2))
  y_all <- rbind(matrix(c(1,0), n/2, 2, byrow = TRUE),
                 matrix(c(0,1), n/2, 2, byrow = TRUE))
  x_cm <- t(x_all)
  y_cm <- t(y_all)

  # "Pretrained" backbone
  backbone <- ag_sequential(
    ag_linear(2L, 16L, activation = "relu"),
    ag_linear(16L, 8L, activation = "relu")
  )
  # New head
  head_layer <- ag_linear(8L, 2L)

  # Snapshot backbone weights before training
  bb_params <- backbone$parameters()
  bb_before <- lapply(bb_params, function(p) p$data)

  # Only optimize head parameters
  head_params <- head_layer$params()
  opt <- optimizer_adam(head_params, lr = 1e-2)

  BS <- 16L
  ag_train(backbone)
  ag_train(head_layer)

  for (ep in seq_len(30L)) {
    perm <- sample(n)
    for (b in seq_len(ceiling(n / BS))) {
      idx <- perm[((b-1L)*BS+1L):min(b*BS, n)]
      xb <- ag_tensor(x_cm[, idx, drop = FALSE])
      yb <- y_cm[, idx, drop = FALSE]
      with_grad_tape({
        features <- backbone$forward(xb)
        logits <- head_layer$forward(features)
        loss <- ag_softmax_cross_entropy_loss(logits, yb)
      })
      grads <- backward(loss)
      # Only step on head params
      opt$step(grads)
      opt$zero_grad()
    }
  }

  # Backbone weights should NOT change (not in optimizer)
  bb_after <- lapply(bb_params, function(p) p$data)
  for (nm in names(bb_before)) {
    expect_equal(bb_before[[nm]], bb_after[[nm]],
                 info = paste("backbone param", nm, "should be frozen"))
  }

  # Head should have updated (loss should be finite)
  ag_eval(backbone)
  ag_eval(head_layer)
  features <- backbone$forward(ag_tensor(x_cm))
  logits <- head_layer$forward(features)$data
  expect_true(all(is.finite(logits)))
})

# ── 3. Sequential save/load with BatchNorm ─────────────────

test_that("chain pattern: save/load sequential+BatchNorm preserves predictions", {
  set.seed(42)
  n <- 60L
  x <- matrix(rnorm(n * 2), n, 2)
  y <- cbind(as.numeric(x[,1] > 0), as.numeric(x[,1] <= 0))

  m <- ggml_model_sequential() |>
    ggml_layer_dense(8L, activation = "relu", input_shape = 2L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  m <- ggml_fit(m, x, y, epochs = 30L, batch_size = 10L, verbose = 0L)
  p_before <- ggml_predict(m, x[1:10, ], batch_size = 10L)

  # Save and load
  path <- tempfile(fileext = ".ggml")
  ggml_save_model(m, path)
  m2 <- ggml_load_model(path)
  p_after <- ggml_predict(m2, x[1:10, ], batch_size = 10L)

  # Predictions should match
  expect_equal(p_before, p_after, tolerance = 1e-4)
})

# ── 4. Functional multi-input: two branches → add → classify ──

test_that("chain pattern: functional multi-input trains and predicts", {
  set.seed(42)
  n <- 80L
  # Branch 1: continuous features
  x1 <- matrix(rnorm(n * 3), n, 3)
  # Branch 2: one-hot categorical (4 categories)
  cats <- sample(1:4, n, replace = TRUE)
  x2 <- matrix(0, n, 4)
  x2[cbind(1:n, cats)] <- 1

  # Target: depends on first feature + category
  y_class <- as.integer(x1[,1] + (cats == 1) > 0)
  y <- cbind(y_class, 1L - y_class) * 1.0

  inp1 <- ggml_input(shape = 3L)
  inp2 <- ggml_input(shape = 4L)
  h1 <- inp1 |> ggml_layer_dense(8L, activation = "relu")
  h2 <- inp2 |> ggml_layer_dense(8L, activation = "relu")
  merged <- ggml_layer_add(list(h1, h2))
  out <- merged |> ggml_layer_dense(2L, activation = "softmax")

  m <- ggml_model(inputs = list(inp1, inp2), outputs = out) |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  m <- ggml_fit(m, x = list(x1, x2), y = y,
                epochs = 50L, batch_size = 10L, verbose = 0L)
  p <- ggml_predict(m, x = list(x1, x2), batch_size = 16L)

  expect_equal(nrow(p), n)
  expect_equal(ncol(p), 2L)
  expect_true(all(is.finite(p)))
  # Probabilities sum to ~1
  expect_true(all(abs(rowSums(p) - 1.0) < 0.05))
})

# ── 5. Sequential: dropout eval is deterministic ──────────

test_that("chain pattern: sequential dropout predict is deterministic", {
  set.seed(42)
  n <- 40L
  x <- matrix(rnorm(n * 2), n, 2)
  y <- cbind(as.numeric(x[,1] > 0), as.numeric(x[,1] <= 0))

  m <- ggml_model_sequential() |>
    ggml_layer_dense(16L, activation = "relu", input_shape = 2L) |>
    ggml_layer_dropout(0.5) |>
    ggml_layer_dense(2L, activation = "softmax") |>
    ggml_compile(optimizer = "adam", loss = "categorical_crossentropy")

  m <- ggml_fit(m, x, y, epochs = 20L, batch_size = 10L, verbose = 0L)

  # Two predict calls should give identical results (eval mode → no dropout)
  p1 <- ggml_predict(m, x[1:10, ], batch_size = 10L)
  p2 <- ggml_predict(m, x[1:10, ], batch_size = 10L)
  expect_equal(p1, p2)
})
