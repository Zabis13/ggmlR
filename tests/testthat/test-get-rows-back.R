# Backward of an embedding lookup: GGML_OP_GET_ROWS_BACK, CPU and Vulkan.
#
# The Vulkan shader is a ggmlR addition. Without it the backward of every
# embedding table ran on the CPU: on a 30000 x 512 vocabulary that measured
# 61.0 ms against 39.1 ms for plain CPU training (0.64x, 7 splits).
#
# The op is a scatter -- repeated tokens accumulate into the same table row --
# so these check the accumulation as well as the arithmetic: a row referenced
# twice must receive both gradients, and a row nobody references must stay zero.

skip_no_vulkan <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

# d(sum(get_rows(table, ids)))/d(table)
embed_grad <- function(tbl_data, id_vec, vocab, dim, backend = c("cpu", "gpu")) {
  backend <- match.arg(backend)

  ctx <- ggml_init(256 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  tbl <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, vocab)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length(id_vec))
  ggml_set_input(tbl); ggml_set_input(ids)
  ggml_set_param(tbl)

  be <- if (backend == "gpu") ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  if (backend == "cpu") ggml_backend_cpu_set_n_threads(be, 2L)
  sched <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit({
    ggml_backend_sched_free(sched)
    if (backend == "gpu") ggml_vulkan_free(be) else ggml_backend_free(be)
  }, add = TRUE)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)

  ggml_backend_tensor_set_data(tbl, tbl_data)
  ggml_backend_tensor_set_data(ids, id_vec)

  loss <- ggml_sum(ctx, ggml_get_rows(ctx, tbl, ids))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_backend_sched_alloc_graph(sched, graph)
  ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  matrix(ggml_backend_tensor_get_data(ggml_graph_get_grad(graph, tbl)), dim, vocab)
}

test_that("embedding gradients count each token's contribution", {
  # d(sum(rows))/d(table) is just the number of times each row was gathered:
  # every element of a referenced row gets exactly its reference count.
  vocab <- 8L; dim <- 4L
  ids <- c(0L, 3L, 3L, 5L)          # row 3 twice, rows 1,2,4,6,7 never
  set.seed(51L)
  tbl <- runif(dim * vocab, -1, 1)

  g <- embed_grad(tbl, ids, vocab, dim, backend = "cpu")

  expect_equal(unname(g[, 1]), rep(1, dim))   # id 0, once
  expect_equal(unname(g[, 4]), rep(2, dim))   # id 3, twice
  expect_equal(unname(g[, 6]), rep(1, dim))   # id 5, once

  untouched <- c(2L, 3L, 5L, 7L, 8L)          # 1-based columns never referenced
  expect_true(all(g[, untouched] == 0))
})

test_that("the Vulkan shader agrees with the CPU kernel, repeats included", {
  skip_no_vulkan()
  vocab <- 64L; dim <- 32L
  set.seed(52L)
  tbl <- runif(dim * vocab, -1, 1)
  # Deliberately many repeats: 128 tokens over 64 rows, so the atomics collide.
  ids <- sample.int(vocab, 128L, replace = TRUE) - 1L

  g_cpu <- embed_grad(tbl, ids, vocab, dim, backend = "cpu")
  g_gpu <- embed_grad(tbl, ids, vocab, dim, backend = "gpu")

  expect_equal(g_gpu, g_cpu, tolerance = 1e-5)

  # Sanity: the totals must equal the token count, whichever backend ran.
  expect_equal(sum(g_gpu) / dim, length(ids))
})

test_that("a table row nobody references keeps a zero gradient on the GPU", {
  # The shader zeroes the destination before accumulating; without that a row
  # no token points at would keep whatever was in the buffer.
  skip_no_vulkan()
  vocab <- 32L; dim <- 8L
  set.seed(53L)
  tbl <- runif(dim * vocab, -1, 1)
  ids <- c(0L, 1L, 2L)              # only the first three rows

  g <- embed_grad(tbl, ids, vocab, dim, backend = "gpu")

  expect_true(all(g[, 4:vocab] == 0))
  expect_equal(unname(g[, 1]), rep(1, dim))
})
