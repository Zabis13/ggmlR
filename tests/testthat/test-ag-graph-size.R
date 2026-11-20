# Graphs sized for their work (.ag_graph_build, R_ggml_graph_count_reachable):
# ggml_new_graph() holds 8192 nodes and an overflow is a GGML_ASSERT, which an
# unrolled loop (an LSTM over 128 steps with its backward) hit in the deferred
# graph, the graph backward and ag_capture_step. Checked: the size at and
# around the old limit, far above it, and the three ag_* paths end to end.

# A chain of n ggml_scale nodes over one leaf: n + 1 reachable tensors.
gs_chain <- function(n) {
  # + 1 MB for the work buffer ggml_graph_compute() takes from the context
  ctx <- ggml_init((n + 8) * (ggml_tensor_overhead() + 64) + 2^20, no_alloc = FALSE)
  x <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1)
  ggml_set_f32(x, 1)
  y <- x
  # 1 + 2^-13 is exact in f32: only the products round, not the factor
  for (i in seq_len(n)) y <- ggml_scale(ctx, y, 1 + 2^-13)
  list(ctx = ctx, x = x, y = y)
}

test_that("graphs at, around and far above 8192 nodes are built and computed", {
  for (n in c(100L, 8190L, 8191L, 8192L, 8193L, 20000L)) {
    ch <- gs_chain(n)
    expect_identical(.Call("R_ggml_graph_count_reachable", list(ch$y), PACKAGE = "ggmlR"),
                     n + 1, info = as.character(n))
    gb <- ggmlR:::.ag_graph_build(list(ch$y), "test graph")
    expect_identical(gb$size, max(8192, n + 1 + 16), info = as.character(n))
    expect_equal(ggml_graph_n_nodes(gb$graph), n, info = as.character(n))
    ggml_graph_compute(ch$ctx, gb$graph)
    expect_equal(ggml_get_f32(ch$y), (1 + 2^-13)^n, tolerance = 1e-3, info = as.character(n))
    ggml_free(gb$ctx); ggml_free(ch$ctx)
  }
  # several roots sharing a part: each tensor counted once
  ch <- gs_chain(10L)
  ctx2 <- ch$ctx
  z <- ggml_scale(ctx2, ch$y, 2)
  expect_identical(.Call("R_ggml_graph_count_reachable", list(ch$y, z, ch$x),
                         PACKAGE = "ggmlR"), 12)
  ggml_free(ctx2)
})

gs_has_gpu <- function() ggml_vulkan_available() && ggml_vulkan_device_count() > 0L

test_that("graph mode, the graph backward and ag_capture_step past 8192 nodes (LSTM, T = 150)", {
  skip_if_not(gs_has_gpu(), "no Vulkan device")
  on.exit(ag_device("cpu"), add = TRUE)
  n_t <- 150L
  run <- function(captured) {
    ag_device("gpu")
    ag_local_mode(graph = TRUE, matmul_precision = "f32")
    set.seed(61)
    layer <- ag_lstm(2L, 3L, init = "torch")
    head <- ag_linear(3L, 1L)
    ps <- c(layer$parameters(), list(hW = head$W, hb = head$b))
    opt <- optimizer_adam(ps, lr = 0.01)
    # the initial state is an input: zeros made inside fn would be host data,
    # which a recording refuses
    d <- list(x = matrix(rnorm(2 * n_t * 2), 2 * n_t, 2), h = matrix(0, 3, 2),
              c = matrix(0, 3, 2), y = matrix(c(0.3, -0.3), 1, 2))
    fn <- function(x, h, c, y) {
      xs <- lapply(seq_len(n_t) - 1L, function(t) ag_slice_rows(x, 2L * t, 2L))
      r <- layer$forward(xs, state = list(h = h, c = c))
      ag_mse_loss(head$forward(r$outputs[[n_t]]), y)
    }
    losses <- numeric(0)
    if (captured) {
      train <- ag_capture_step(fn, optimizer = opt)
      on.exit(ag_capture_free(train), add = TRUE)
      for (k in 1:2) losses <- c(losses, as.numeric(do.call(train, d)))
    } else {
      args <- lapply(d, ag_tensor)
      for (k in 1:2) {
        loss <- NULL
        with_grad_tape(loss <- do.call(fn, args))
        opt$step(backward(loss)); opt$zero_grad()
        losses <- c(losses, as.numeric(as.matrix(loss)))
      }
    }
    list(losses = losses, w = lapply(ps, as.matrix))
  }
  ref <- run(FALSE)                                   # deferred graph + graph backward
  got <- run(TRUE)                                    # one recording
  expect_true(all(is.finite(ref$losses)))
  expect_equal(got$losses, ref$losses, tolerance = 1e-5)
  expect_equal(got$w, ref$w, tolerance = 1e-5)
})
