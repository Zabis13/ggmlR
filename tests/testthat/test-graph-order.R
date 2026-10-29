# ggml_graph_order (ggml 0.27 ggml_build_forward_order) and the compute flag.
#
# A node added with ggml_graph_order is placed in the graph but not run until a
# node added with ggml_graph_expand reaches it. Backends skip nodes without the
# compute flag: the Vulkan backend always did, the CPU loop now does too (as in
# upstream). ggml_graph_add_node marks its node itself, so it runs on both.

SENTINEL <- -12345

# a -> b = 2a (ordered only) ; c = a + 1 (expanded) ; optionally expand(b) later
run_order <- function(use_gpu, expand_b) {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4); ggml_set_name(a, "a")
  b <- ggml_scale(ctx, a, 2);                     ggml_set_name(b, "b")
  c <- ggml_scale(ctx, a, 3);                     ggml_set_name(c, "c")
  graph <- ggml_build_forward_expand(ctx, a)   # leaf only: a graph with no nodes yet
  ggml_graph_order(graph, b)
  ggml_graph_expand(graph, c)
  if (expand_b) ggml_graph_expand(graph, b)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(a, c(1, 2, 3, 4))
  ggml_backend_tensor_set_data(b, rep(SENTINEL, 4))
  ggml_backend_graph_compute(backend, graph)

  names <- vapply(seq_len(ggml_graph_n_nodes(graph)) - 1L,
                  function(i) ggml_get_name(ggml_graph_node(graph, i)), "")
  list(b = ggml_backend_tensor_get_data(b), c = ggml_backend_tensor_get_data(c), order = names)
}

for (use_gpu in c(FALSE, TRUE)) {
  local({
    gpu <- use_gpu
    lbl <- if (gpu) "Vulkan" else "CPU"
    test_that(sprintf("%s: a node only ordered is placed but not computed", lbl), {
      if (gpu) skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      r <- run_order(gpu, expand_b = FALSE)
      expect_identical(r$order, c("b", "c"))       # b keeps its place ahead of c
      expect_equal(r$b, rep(SENTINEL, 4))          # ... but was not run
      expect_equal(r$c, c(3, 6, 9, 12))
    })
    test_that(sprintf("%s: expanding an ordered node later makes it compute", lbl), {
      if (gpu) skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      r <- run_order(gpu, expand_b = TRUE)
      expect_identical(r$order, c("b", "c"))       # still in the ordered position
      expect_equal(r$b, c(2, 4, 6, 8))
      expect_equal(r$c, c(3, 6, 9, 12))
    })
    test_that(sprintf("%s: ggml_graph_add_node marks its node for compute", lbl), {
      if (gpu) skip_if_not(ggml_vulkan_available(), "no Vulkan device")
      ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
      on.exit(ggml_free(ctx))
      a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3)
      d <- ggml_scale(ctx, a, 5)
      graph <- ggml_build_forward_expand(ctx, a)   # leaf only, no nodes
      ggml_graph_add_node(graph, d)
      backend <- if (gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
      on.exit(ggml_backend_free(backend), add = TRUE)
      buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
      on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
      ggml_backend_tensor_set_data(a, c(1, 2, 3))
      ggml_backend_graph_compute(backend, graph)
      expect_equal(ggml_backend_tensor_get_data(d), c(5, 10, 15))
    })
  })
}

test_that("ggml_is_contiguous_to_N checks only the leading dims", {
  ctx <- ggml_init(1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))
  x <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 6, 5, 4)
  expect_true(all(ggml_is_contiguous_to_1(x), ggml_is_contiguous_to_2(x), ggml_is_contiguous_to_3(x)))

  # rows of 4 out of 6: dim 0 packed, dim 1 strided
  v <- ggml_view_3d(ctx, x, 4, 5, 4, 6 * 4, 6 * 5 * 4, 0)
  expect_true(ggml_is_contiguous_to_1(v))
  expect_false(ggml_is_contiguous_to_2(v))

  # dims 1 and 2 swapped: dim 0 packed, dim 1 now steps over whole planes
  p <- ggml_permute(ctx, x, 0, 2, 1, 3)
  expect_true(ggml_is_contiguous_to_1(p))
  expect_false(ggml_is_contiguous_to_2(p))

  # planes padded: dims 0..1 packed, dim 2 strided
  w <- ggml_view_3d(ctx, x, 6, 4, 4, 6 * 4, 6 * 5 * 4, 0)
  expect_true(ggml_is_contiguous_to_2(w))
  expect_false(ggml_is_contiguous_to_3(w))
})
