# ag_reshape() and ag_layer_norm() as tape nodes the graph backward can emit
# ("reshape": one reshape node; "layer_norm": one ggml_norm_back node), so a
# DreamerV3-style world model (categorical latents regrouped with ag_reshape,
# LayerNorm everywhere) can be an ag_capture_step.

rl_tape_grads <- function(X, W, G, graph) {
  old <- ggmlR:::ag_backward_graph(graph)
  withr::defer(ggmlR:::ag_backward_graph(old))
  ln <- ag_layer_norm(nrow(W))
  x  <- ag_param(X); w <- ag_param(W)
  # [8, 6] -> [4, 12]: softmax over groups of 4, then back to [8, 6]
  with_grad_tape({
    h    <- ln$forward(ag_matmul(w, x))
    p    <- ag_softmax(ag_reshape(h, 4L, 2L * ncol(X)))
    y    <- ag_reshape(p, nrow(W), ncol(X))
    loss <- ag_sum(ag_mul(y, ag_tensor(G)))
  })
  backward(loss)
  list(path  = ggmlR:::ag_backward_path(),
       x     = ggmlR:::.ag_as_matrix(x$grad),
       w     = ggmlR:::.ag_as_matrix(w$grad),
       gamma = ggmlR:::.ag_as_matrix(ln$gamma$grad),
       beta  = ggmlR:::.ag_as_matrix(ln$beta$grad))
}

rl_data <- function() {
  set.seed(5L)
  list(X = matrix(rnorm(6 * 6), 6, 6), W = matrix(rnorm(8 * 6), 8, 6),
       G = matrix(rnorm(8 * 6), 8, 6))
}

test_that("reshape and layer_norm gradients on the CPU match finite differences", {
  local_cpu_device()
  d <- rl_data()
  f <- function(X) {
    ln <- ag_layer_norm(8L)
    y  <- ag_reshape(ag_softmax(ag_reshape(ln$forward(ag_matmul(ag_tensor(d$W), ag_tensor(X))),
                                           4L, 12L)), 8L, 6L)
    sum(as.matrix(y) * d$G)
  }
  g <- rl_tape_grads(d$X, d$W, d$G, graph = FALSE)$x
  num <- matrix(0, nrow(d$X), ncol(d$X))
  for (i in seq_along(d$X)) {
    Xp <- d$X; Xp[i] <- Xp[i] + 1e-5
    Xm <- d$X; Xm[i] <- Xm[i] - 1e-5
    num[i] <- (f(Xp) - f(Xm)) / 2e-5
  }
  expect_equal(g, num, tolerance = 1e-5)
})

test_that("reshape and layer_norm keep a tape on the graph backward with the closure gradients", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")

  d <- rl_data()
  g_graph <- rl_tape_grads(d$X, d$W, d$G, graph = TRUE)
  g_clos  <- rl_tape_grads(d$X, d$W, d$G, graph = FALSE)

  expect_identical(g_graph$path, "graph")
  for (nm in c("x", "w", "gamma", "beta"))
    expect_equal(g_graph[[nm]], g_clos[[nm]], tolerance = 1e-4, info = nm)
})

test_that("layer_norm forward on the GPU matches the CPU", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  set.seed(6L)
  X <- matrix(rnorm(16 * 5) * 3 + 2, 16, 5)
  ref <- as.matrix(ag_layer_norm(16L)$forward(ag_tensor(X)))
  ag_device("gpu"); ag_dtype("f32")
  got <- as.matrix(ag_layer_norm(16L)$forward(ag_tensor(X)))
  expect_equal(got, ref, tolerance = 1e-5)
})

test_that("ag_capture_step records layer_norm and reshape and follows the uncaptured loop", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()

  run <- function(captured) {
    ag_device("gpu"); ag_dtype("f32")
    ag_local_mode(graph = TRUE, matmul_precision = "f32")
    set.seed(7L)
    W  <- ag_param(matrix(rnorm(8 * 6) * 0.3, 8, 6))
    ln <- ag_layer_norm(8L)
    ps <- c(list(W = W), ln$parameters())
    X  <- matrix(rnorm(6 * 10), 6, 10); Y <- matrix(runif(8 * 10), 8, 10)
    fn <- function(x, y) {
      p <- ag_softmax(ag_reshape(ln$forward(ag_matmul(W, x)), 4L, 20L))
      ag_mse_loss(ag_reshape(p, 8L, 10L), y)
    }
    opt <- optimizer_adam(ps, lr = 0.05)
    losses <- numeric(0)
    if (captured) {
      tr <- ag_capture_step(fn, optimizer = opt)
      on.exit(ag_capture_free(tr), add = TRUE)
      for (k in 1:5) losses <- c(losses, as.numeric(tr(x = X, y = Y)))
    } else {
      x <- ag_tensor(X); y <- ag_tensor(Y)
      for (k in 1:5) {
        loss <- NULL
        with_grad_tape(loss <- fn(x, y))
        opt$step(backward(loss)); opt$zero_grad()
        losses <- c(losses, as.numeric(as.matrix(loss)))
      }
    }
    losses
  }
  ref <- run(FALSE)
  got <- run(TRUE)
  expect_equal(got, ref, tolerance = 1e-4)
  expect_lt(got[5], got[1])
})
