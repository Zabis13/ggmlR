# ag_flash_attention(batch = B): B independent sequences in one fused call.
#
# The batch is ggml's fourth axis; in R it is B sequences laid side by side in
# the columns. The reference for every check here is the SAME function called
# once per sample -- so a batched call that leaked attention across samples,
# or cut the columns at the wrong place, disagrees with it.

ag_flash_attention <- ggmlR:::ag_flash_attention

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

cols_of <- function(b, s) ((b - 1L) * s + 1L):(b * s)

# Leaf gradients of sum(G * attention) for one call, as plain matrices.
grads_of <- function(Q, K, V, G, h, batch = 1L, causal = FALSE) {
  qq <- ag_param(Q); kk <- ag_param(K); vv <- ag_param(V)
  with_grad_tape({
    out  <- ag_flash_attention(qq, kk, vv, n_heads = h, batch = batch,
                               causal = causal)
    loss <- ag_sum(ag_mul(out, ag_tensor(G)))
  })
  backward(loss)
  list(out = ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(out)),
       q = ggmlR:::.ag_as_matrix(qq$grad),
       k = ggmlR:::.ag_as_matrix(kk$grad),
       v = ggmlR:::.ag_as_matrix(vv$grad))
}

# The same quantities assembled from one unbatched call per sample.
grads_per_sample <- function(Q, K, V, G, h, B, s, causal = FALSE) {
  r <- list(out = Q * 0, q = Q * 0, k = K * 0, v = V * 0)
  for (b in seq_len(B)) {
    cc <- cols_of(b, s)
    g1 <- grads_of(Q[, cc, drop = FALSE], K[, cc, drop = FALSE],
                   V[, cc, drop = FALSE], G[, cc, drop = FALSE], h,
                   causal = causal)
    for (nm in names(r)) r[[nm]][, cc] <- g1[[nm]]
  }
  r
}

mk <- function(d, n) matrix(runif(d * n, -1, 1), d, n)

test_that("batched attention equals one call per sample (CPU)", {
  with_cpu_backend({
    for (causal in c(FALSE, TRUE)) {
      d <- 8L; h <- 2L; s <- 4L; B <- 3L
      set.seed(11L)
      Q <- mk(d, B * s); K <- mk(d, B * s); V <- mk(d, B * s); G <- mk(d, B * s)

      got <- grads_of(Q, K, V, G, h, batch = B, causal = causal)
      ref <- grads_per_sample(Q, K, V, G, h, B, s, causal = causal)

      expect_equal(dim(got$out), c(d, B * s))
      for (nm in names(ref))
        expect_equal(got[[nm]], ref[[nm]], tolerance = 1e-5,
                     info = paste(nm, "causal", causal))
    }
  })
})

test_that("batch = 1 is the old behaviour", {
  with_cpu_backend({
    d <- 8L; h <- 2L; s <- 5L
    set.seed(3L)
    Q <- mk(d, s); K <- mk(d, s); V <- mk(d, s)
    a <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h)
    b <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h,
                            batch = 1L)
    expect_identical(ggmlR:::.ag_data(a), ggmlR:::.ag_data(b))
  })
})

test_that("batch must divide the column counts", {
  with_cpu_backend({
    Q <- mk(8L, 10L)
    expect_error(ag_flash_attention(ag_tensor(Q), ag_tensor(Q), ag_tensor(Q),
                                    2L, batch = 3L),
                 "multiples of batch")
    expect_error(ag_flash_attention(ag_tensor(Q), ag_tensor(Q), ag_tensor(Q),
                                    2L, batch = 0L),
                 "positive whole")
  })
})

# ---------------------------------------------------------------- GPU, graph

# Vulkan's flash attention forward multiplies in f16 on cooperative-matrix
# devices whatever precision is requested (2.9e-4 / 7.0e-4 max abs off the CPU
# on an RX 9070, RADV), so `out` gets an f16-sized tolerance. The gradients come
# from our own f32 ggml_flash_attn_back and keep the tight one.
gpu_tol <- c(out = 2e-3, q = 1e-4, k = 1e-4, v = 1e-4)

gpu_graph_grads <- function(Q, K, V, G, h, B) {
  old <- ggmlR:::ag_backward_graph(TRUE)
  withr::defer(ggmlR:::ag_backward_graph(old))
  r <- grads_of(Q, K, V, G, h, batch = B)
  r$path <- ggmlR:::ag_backward_path()
  r
}

test_that("a batched attention tape runs on the graph backward and matches the CPU", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()

  d <- 8L; h <- 2L; s <- 4L; B <- 5L
  set.seed(5L)
  Q <- mk(d, B * s); K <- mk(d, B * s); V <- mk(d, B * s); G <- mk(d, B * s)

  ref <- with_cpu_backend(grads_per_sample(Q, K, V, G, h, B, s))

  ag_device("gpu"); ag_dtype("f32")
  got <- gpu_graph_grads(Q, K, V, G, h, B)

  expect_identical(got$path, "graph")
  for (nm in names(gpu_tol))
    expect_equal(got[[nm]], ref[[nm]], tolerance = gpu_tol[[nm]], info = nm)
})

test_that("more than 65535 query rows spill into a second dispatch axis", {
  # N * H * B = 4 * 2 * 8200 = 65600 workgroups for the backward: past the
  # smallest maxComputeWorkGroupCount[0] Vulkan allows, so the rows have to be
  # laid over x and y. Before, this was 128x as many on x and an abort on any
  # device whose x limit is 65535.
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  skip_on_cran()
  local_cpu_device()

  d <- 8L; h <- 2L; s <- 4L; B <- 8200L
  set.seed(9L)
  Q <- mk(d, B * s); K <- mk(d, B * s); V <- mk(d, B * s); G <- mk(d, B * s)

  ref <- with_cpu_backend(grads_of(Q, K, V, G, h, batch = B))

  ag_device("gpu"); ag_dtype("f32")
  got <- gpu_graph_grads(Q, K, V, G, h, B)

  expect_identical(got$path, "graph")
  for (nm in names(gpu_tol))
    expect_equal(got[[nm]], ref[[nm]], tolerance = gpu_tol[[nm]], info = nm)
  # The last samples are the ones a truncated dispatch would leave at zero.
  last <- cols_of(B, s)
  expect_gt(max(abs(got$q[, last])), 0)
})

test_that("ag_matmul_precision('f32') reaches the attention forward on both paths", {
  # The flag asks for f32 accumulation; how much that buys depends on the
  # device (nothing on the coopmat path, where the operands are f16 anyway).
  # What can be pinned down everywhere: the eager path (no tape) and the
  # resident one (on a tape) build the same node and agree with each other,
  # and both stay within f16 accuracy of the CPU.
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()

  d <- 16L; h <- 4L; s <- 6L; B <- 4L
  set.seed(21L)
  Q <- mk(d, B * s); K <- mk(d, B * s); V <- mk(d, B * s); G <- mk(d, B * s)
  ref <- with_cpu_backend(grads_of(Q, K, V, G, h, batch = B))$out

  ag_device("gpu"); ag_dtype("f32")
  old_p <- ag_matmul_precision("f32")
  withr::defer(ag_matmul_precision(old_p))

  eager <- ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(
    ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h, batch = B)))
  taped <- gpu_graph_grads(Q, K, V, G, h, B)$out

  expect_equal(eager, taped, tolerance = 1e-6)
  expect_equal(eager, ref, tolerance = gpu_tol[["out"]])
})
