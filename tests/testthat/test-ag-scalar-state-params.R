# A. ag_add / ag_sub with a plain number: CPU and GPU agree, backward runs (on
#    the GPU one ggml_scale_bias node, recordable by ag_capture); a bare vector
#    is a clear error on the GPU.
# B. GRU / LSTM state = NULL: zeros made on the input's device, B from the
#    input; an ag_capture recording needs no h0.
# C. ag_linear: $parameters() is the method, $params() the same function.

skip_no_vk <- function() {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
}

# ---- A ----------------------------------------------------------------------

sc_cases <- list(
  list(f = function(x) ag_add(x, 0.5),  val = function(X) X + 0.5,  d = 1),
  list(f = function(x) ag_add(-2, x),   val = function(X) -2 + X,   d = 1),
  list(f = function(x) ag_sub(x, 1.25), val = function(X) X - 1.25, d = 1),
  list(f = function(x) ag_sub(3, x),    val = function(X) 3 - X,    d = -1))

sc_run <- function(cs, X, G) {
  x <- ag_param(X)
  with_grad_tape({
    y    <- cs$f(x)
    loss <- ag_sum(ag_mul(y, ag_tensor(G)))
  })
  backward(loss)
  list(y = as.matrix(y), g = ggmlR:::.ag_as_matrix(x$grad),
       path = ggmlR:::ag_backward_path())
}

test_that("ag_add / ag_sub with a number on either side: values and gradient (CPU)", {
  local_cpu_device()
  set.seed(1L)
  X <- matrix(rnorm(12), 3, 4); G <- matrix(rnorm(12), 3, 4)
  for (i in seq_along(sc_cases)) {
    r <- sc_run(sc_cases[[i]], X, G)
    expect_equal(r$y, sc_cases[[i]]$val(X), info = as.character(i))
    expect_equal(r$g, G * sc_cases[[i]]$d, info = as.character(i))
  }
})

test_that("ag_add / ag_sub with a number on the GPU match the CPU, graph backward", {
  skip_no_vk()
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")
  set.seed(2L)
  X <- matrix(rnorm(12), 3, 4); G <- matrix(rnorm(12), 3, 4)
  for (i in seq_along(sc_cases)) {
    r <- sc_run(sc_cases[[i]], X, G)
    expect_equal(r$y, sc_cases[[i]]$val(X), tolerance = 1e-6, info = as.character(i))
    expect_equal(r$g, G * sc_cases[[i]]$d, tolerance = 1e-6, info = as.character(i))
    expect_identical(r$path, "graph", info = as.character(i))
  }
})

test_that("a number in ag_add / ag_sub is recordable by ag_capture (no upload)", {
  skip_no_vk()
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")
  f <- ag_capture(function(x) ag_sub(ag_add(ag_scale(x, 0.99), 0.01 / 8), 1))
  on.exit(ag_capture_free(f), add = TRUE)
  X <- matrix(runif(16), 8, 2)
  expect_equal(f(x = X), X * 0.99 + 0.01 / 8 - 1, tolerance = 1e-6)
  expect_equal(f(x = 2 * X), 2 * X * 0.99 + 0.01 / 8 - 1, tolerance = 1e-6)
})

test_that("a bare vector in ag_add / ag_sub on the GPU is a clear error", {
  skip_no_vk()
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")
  x <- ag_tensor(matrix(1, 3, 4))
  expect_error(ag_add(x, c(1, 2, 3)), "vector of length 3")
  expect_error(ag_sub(c(1, 2, 3, 4), x), "vector of length 4")
})

# ---- B ----------------------------------------------------------------------

st_run <- function(kind, X, explicit_zero) {
  set.seed(5L)
  cell <- if (kind == "gru") ag_gru_cell(3L, 4L) else ag_lstm_cell(3L, 4L)
  x <- ag_param(X); B <- ncol(X)
  s0 <- if (explicit_zero) {
    z <- ag_tensor(matrix(0, 4L, B))
    if (kind == "gru") list(h = z) else list(h = z, c = ag_tensor(matrix(0, 4L, B)))
  }
  with_grad_tape({
    s    <- cell$forward(x, s0)
    s    <- cell$forward(x, s)
    loss <- ag_sum(ag_mul(s$h, s$h))
  })
  backward(loss)
  list(h = as.matrix(s$h), gx = ggmlR:::.ag_as_matrix(x$grad),
       gw = ggmlR:::.ag_as_matrix(cell$weight_hh$grad))
}

test_that("RNN state = NULL equals an explicit zero state on the GPU (GRU, LSTM)", {
  skip_no_vk()
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")
  set.seed(6L)
  for (kind in c("gru", "lstm")) for (B in c(2L, 5L)) {      # B changes between calls
    X <- matrix(rnorm(3 * B), 3, B)
    a <- st_run(kind, X, explicit_zero = FALSE)
    b <- st_run(kind, X, explicit_zero = TRUE)
    for (nm in c("h", "gx", "gw"))
      expect_equal(a[[nm]], b[[nm]], tolerance = 1e-6, info = paste(kind, B, nm))
    expect_identical(dim(a$h), c(4L, B))
  }
})

test_that("an ag_capture_step with state = NULL needs no h0 and follows the loop", {
  skip_no_vk()
  local_cpu_device()
  for (kind in c("gru", "lstm")) {
    run <- function(captured) {
      ag_device("gpu"); ag_dtype("f32")
      ag_local_mode(graph = TRUE, matmul_precision = "f32")
      set.seed(7L)
      cell <- if (kind == "gru") ag_gru_cell(3L, 4L) else ag_lstm_cell(3L, 4L)
      head <- ag_linear(4L, 1L)
      X <- matrix(rnorm(3 * 6), 3, 6); Y <- matrix(rnorm(6), 1, 6)
      fn <- function(x, y) {
        s <- cell$forward(x)                    # no h0: zeros on the device
        s <- cell$forward(x, s)
        ag_mse_loss(head$forward(s$h), y)
      }
      opt <- optimizer_adam(c(cell$parameters(), head$parameters()), lr = 0.05)
      losses <- numeric(0)
      if (captured) {
        tr <- ag_capture_step(fn, optimizer = opt)
        on.exit(ag_capture_free(tr), add = TRUE)
        for (k in 1:4) losses <- c(losses, as.numeric(tr(x = X, y = Y)))
      } else {
        x <- ag_tensor(X); y <- ag_tensor(Y)
        for (k in 1:4) {
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
    expect_equal(got, ref, tolerance = 1e-4, info = kind)
  }
})

# ---- C ----------------------------------------------------------------------

test_that("ag_linear: parameters() and params() are the same list in the same order", {
  local_cpu_device()
  l <- ag_linear(3L, 2L)
  expect_true(is.function(l$parameters))
  expect_identical(l$parameters, l$params)
  p1 <- l$parameters(); p2 <- l$params()
  expect_identical(names(p1), c("W", "b"))
  expect_identical(names(p2), names(p1))
  for (nm in names(p1)) expect_identical(p1[[nm]]$id, p2[[nm]]$id)
})

test_that("ag_sequential: parameter order is stable and built from parameters()", {
  local_cpu_device()
  set.seed(8L)
  m <- ag_sequential(ag_linear(3L, 4L), ag_layer_norm(4L), ag_linear(4L, 2L))
  a <- m$parameters(); b <- m$parameters()
  expect_identical(names(a), names(b))
  expect_identical(vapply(a, function(p) p$id, numeric(1)),
                   vapply(b, function(p) p$id, numeric(1)))
  expect_length(a, 6L)
  # an optimizer gets the same tensors whichever name is used
  l <- ag_linear(3L, 2L)
  expect_identical(lapply(optimizer_adam(l$params())$params, `[[`, "id"),
                   lapply(optimizer_adam(l$parameters())$params, `[[`, "id"))
})
