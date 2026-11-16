# ag_silu() and ag_rms_norm(): the activation and the normalisation of
# ggml_layer_transformer_block, as single tape nodes that the graph backward
# can emit -- so a transformer built from ag_* can be an ag_capture_step.

with_cpu_backend <- function(code) {
  st <- ggmlR:::.ag_device_state
  prev_dev <- st$device
  prev_bk  <- st$backend
  on.exit({
    st$device  <- prev_dev
    st$backend <- prev_bk
  }, add = TRUE)
  st$device  <- "cpu"
  st$backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(st$backend, ggml_get_n_threads())
  eval(substitute(code), envir = parent.frame())
}

silu_ref <- function(x) x / (1 + exp(-x))

rms_ref <- function(x, gamma, beta, eps = 1e-5) {
  r <- sqrt(colMeans(x^2) + eps)
  (x / rep(r, each = nrow(x))) * gamma + beta
}

num_grad <- function(f, X, eps = 1e-6) {
  g <- numeric(length(X))
  for (i in seq_along(g)) {
    hi <- X; hi[i] <- hi[i] + eps
    lo <- X; lo[i] <- lo[i] - eps
    g[i] <- (f(hi) - f(lo)) / (2 * eps)
  }
  matrix(g, nrow(X), ncol(X))
}

test_that("ag_silu matches x * sigmoid(x) and its gradient (CPU)", {
  with_cpu_backend({
    set.seed(1L)
    X <- matrix(rnorm(12, sd = 2), 4, 3); G <- matrix(rnorm(12), 4, 3)
    x <- ag_param(X)
    with_grad_tape({
      y    <- ag_silu(x)
      loss <- ag_sum(ag_mul(y, ag_tensor(G)))
    })
    backward(loss)
    expect_equal(ggmlR:::.ag_data(y), silu_ref(X), tolerance = 1e-12)
    expect_equal(ggmlR:::.ag_as_matrix(x$grad),
                 num_grad(function(z) sum(G * silu_ref(z)), X),
                 tolerance = 1e-6)
  })
})

test_that("ag_rms_norm matches its formula and finite differences (CPU)", {
  with_cpu_backend({
    set.seed(2L)
    d <- 5L; n <- 3L
    X <- matrix(rnorm(d * n), d, n); G <- matrix(rnorm(d * n), d, n)
    rn <- ag_rms_norm(d)
    gamma <- matrix(runif(d, 0.5, 1.5), d, 1L); beta <- matrix(rnorm(d), d, 1L)
    rn$gamma <- ag_param(gamma); rn$beta <- ag_param(beta)

    x <- ag_param(X)
    with_grad_tape({
      y    <- rn$forward(x)
      loss <- ag_sum(ag_mul(y, ag_tensor(G)))
    })
    backward(loss)

    expect_equal(ggmlR:::.ag_data(y), rms_ref(X, as.numeric(gamma), as.numeric(beta)),
                 tolerance = 1e-10)
    expect_equal(ggmlR:::.ag_as_matrix(x$grad),
                 num_grad(function(z) sum(G * rms_ref(z, as.numeric(gamma),
                                                      as.numeric(beta))), X),
                 tolerance = 1e-6)
    expect_equal(as.numeric(ggmlR:::.ag_as_matrix(rn$gamma$grad)),
                 rowSums(G * rms_ref(X, 1, 0)), tolerance = 1e-8)
    expect_equal(as.numeric(ggmlR:::.ag_as_matrix(rn$beta$grad)),
                 rowSums(G), tolerance = 1e-10)
  })
})

test_that("ag_rms_norm checks its feature count and lists its parameters", {
  rn <- ag_rms_norm(4L)
  expect_named(rn$parameters(), c("gamma", "beta"))
  expect_length(ag_rms_norm(4L, elementwise_affine = FALSE)$parameters(), 0L)
  expect_error(rn$forward(ag_tensor(matrix(0, 3, 2))), "expected 4 features")
})

# ---------------------------------------------------------------- GPU, graph

# silu and rms_norm on one tape, with a linear layer around them so the tape
# also carries the ops they have to coexist with.
tape_grads <- function(X, W, G, graph) {
  old <- ggmlR:::ag_backward_graph(graph)
  withr::defer(ggmlR:::ag_backward_graph(old))
  rn <- ag_rms_norm(nrow(W))
  x  <- ag_param(X); w <- ag_param(W)
  with_grad_tape({
    h    <- rn$forward(ag_matmul(w, x))
    y    <- ag_silu(h)
    loss <- ag_sum(ag_mul(y, ag_tensor(G)))
  })
  backward(loss)
  list(path  = ggmlR:::ag_backward_path(),
       x     = ggmlR:::.ag_as_matrix(x$grad),
       w     = ggmlR:::.ag_as_matrix(w$grad),
       gamma = ggmlR:::.ag_as_matrix(rn$gamma$grad),
       beta  = ggmlR:::.ag_as_matrix(rn$beta$grad))
}

test_that("silu and rms_norm keep a tape on the graph backward with the closure gradients", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")

  set.seed(4L)
  X <- matrix(rnorm(6 * 7), 6, 7); W <- matrix(rnorm(8 * 6), 8, 6)
  G <- matrix(rnorm(8 * 7), 8, 7)

  g_graph <- tape_grads(X, W, G, graph = TRUE)
  g_clos  <- tape_grads(X, W, G, graph = FALSE)

  expect_identical(g_graph$path, "graph")
  for (nm in c("x", "w", "gamma", "beta"))
    expect_equal(g_graph[[nm]], g_clos[[nm]], tolerance = 1e-4, info = nm)
})
