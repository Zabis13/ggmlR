# Fused bias gradient in the graph backward (R/ag_backward_graph.R): a dense
# layer add(matmul(W, x), b) gets dW and db from one ggml_out_prod_rowsum node
# instead of an out_prod and a separate bias reduction. Checked against the
# unfused graph backward (options(ggmlR.bias_grad_fuse = FALSE)) and against
# the closure backward, with non-zero biases so an error in db cannot hide
# behind zeros, and for bit-identical repeats.

ag_backward_graph <- ggmlR:::ag_backward_graph
ag_backward_path  <- ggmlR:::ag_backward_path
.bf_as_matrix     <- get(".ag_as_matrix", envir = asNamespace("ggmlR"))
.bf_state         <- get(".ag_bwd", envir = asNamespace("ggmlR"))

skip_if_no_gpu_bf <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

# Two dense layers (features x batch, as ag_linear), silu between (in the
# graph backward's op list), MSE loss.
# Returns the gradients of W1, b1, W2, b2 as host matrices.
mlp_grads <- function(graph = TRUE, fuse = TRUE, n_in = 9L, n_h = 64L, n_out = 9L,
                      batch = 512L, seed = 3L) {
  old_g <- ag_backward_graph(graph)
  old_f <- options(ggmlR.bias_grad_fuse = fuse)
  on.exit({ ag_backward_graph(old_g %||% FALSE); options(old_f) }, add = TRUE)
  set.seed(seed)
  W1 <- ag_param(matrix(rnorm(n_h * n_in, sd = 0.3), n_h, n_in))
  b1 <- ag_param(matrix(rnorm(n_h, mean = 0.5), n_h, 1L))         # non-zero
  W2 <- ag_param(matrix(rnorm(n_out * n_h, sd = 0.3), n_out, n_h))
  b2 <- ag_param(matrix(rnorm(n_out, mean = -0.5), n_out, 1L))
  x <- ag_tensor(matrix(rnorm(n_in * batch), n_in, batch))
  y <- ag_tensor(matrix(rnorm(n_out * batch), n_out, batch))
  loss <- NULL
  with_grad_tape({
    h <- ag_silu(ag_add(ag_matmul(W1, x), b1))
    o <- ag_add(ag_matmul(W2, h), b2)
    loss <- ag_mse_loss(o, y)
  })
  .bf_state$fused_bias <- 0L
  backward(loss)
  list(path = ag_backward_path(), fused = .bf_state$fused_bias,
       g = lapply(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2),
                  function(p) .bf_as_matrix(p$grad)))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

rel <- function(a, b) max(abs(a - b)) / max(abs(b))

test_that("fused bias gradient equals the unfused graph backward", {
  skip_if_no_gpu_bf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_matmul_precision("f32"); on.exit(ag_matmul_precision(old), add = TRUE)
  f <- mlp_grads(fuse = TRUE)
  u <- mlp_grads(fuse = FALSE)
  expect_identical(f$path, "graph")
  expect_identical(u$path, "graph")
  expect_identical(f$fused, 2L)                # both layers fused
  expect_identical(u$fused, 0L)
  for (nm in names(f$g)) expect_lt(rel(f$g[[nm]], u$g[[nm]]), 1e-5, label = nm)
  # the biases really have gradients to compare
  expect_gt(max(abs(f$g$b1)), 1e-4)
  expect_gt(max(abs(f$g$b2)), 1e-4)
})

test_that("fused bias gradient equals the closure backward", {
  skip_if_no_gpu_bf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old <- ag_matmul_precision("f32"); on.exit(ag_matmul_precision(old), add = TRUE)
  f <- mlp_grads(fuse = TRUE)
  c <- mlp_grads(graph = FALSE)
  expect_identical(c$path, "closures")
  for (nm in names(f$g)) expect_lt(rel(f$g[[nm]], c$g[[nm]]), 1e-4, label = nm)
})

test_that("fused bias gradient is bit-identical across repeats", {
  skip_if_no_gpu_bf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  a <- mlp_grads(fuse = TRUE, n_h = 256L, batch = 2048L)
  b <- mlp_grads(fuse = TRUE, n_h = 256L, batch = 2048L)
  expect_identical(a$g, b$g)
})

test_that("no fusion when the matmul output has another consumer", {
  skip_if_no_gpu_bf()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  old_g <- ag_backward_graph(TRUE); on.exit(ag_backward_graph(old_g %||% FALSE), add = TRUE)
  set.seed(5)
  W <- ag_param(matrix(rnorm(16 * 9), 16, 9))
  b <- ag_param(matrix(rnorm(16, 1), 16, 1L))
  x <- ag_tensor(matrix(rnorm(9 * 64), 9, 64))
  y <- ag_tensor(matrix(rnorm(16 * 64), 16, 64))
  loss <- NULL
  with_grad_tape({
    z <- ag_matmul(W, x)
    o <- ag_add(ag_add(z, b), z)              # z feeds two nodes: g differs
    loss <- ag_mse_loss(o, y)
  })
  .bf_state$fused_bias <- 0L
  g <- backward(loss)
  expect_identical(ag_backward_path(), "graph")
  expect_identical(.bf_state$fused_bias, 0L)
  # db is still right: compare with the closures
  ag_backward_graph(FALSE)
  set.seed(5)
  W2 <- ag_param(matrix(rnorm(16 * 9), 16, 9))
  b2 <- ag_param(matrix(rnorm(16, 1), 16, 1L))
  x2 <- ag_tensor(matrix(rnorm(9 * 64), 9, 64))
  y2 <- ag_tensor(matrix(rnorm(16 * 64), 16, 64))
  with_grad_tape({
    z2 <- ag_matmul(W2, x2)
    loss2 <- ag_mse_loss(ag_add(ag_add(z2, b2), z2), y2)
  })
  backward(loss2)
  expect_lt(rel(.bf_as_matrix(b$grad), .bf_as_matrix(b2$grad)), 1e-3)
})
