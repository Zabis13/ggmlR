# ggml-alloc must not reuse the memory behind an OUTPUT view.
#
# ggml_clamp returns a view of its input, written in place. The output flag sits
# on that view, but the allocator used to check only the view_src's flag when
# the view's consumers were done -- so it freed the view_src and handed the
# memory to the next node of the same size, overwriting the output. Found as a
# "Vulkan race" in rltoolsR (done flags of a tic-tac-toe step), but it is
# backend-independent: the CPU shows it too.

alloc_view_out <- function(be) {
  cs <- ggml_init(1e6, no_alloc = TRUE)
  on.exit(ggml_free(cs), add = TRUE, after = FALSE)
  x <- ggml_new_tensor_1d(cs, GGML_TYPE_F32, 8)
  buf <- ggml_backend_alloc_ctx_tensors(cs, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  ggml_backend_tensor_set_data(x, as.numeric(1:8))

  g <- ggml_init(1e7, no_alloc = TRUE)
  on.exit(ggml_free(g), add = TRUE, after = FALSE)
  y  <- ggml_add(g, x, x)               # 2x
  v  <- ggml_clamp(g, y, 0, 100)        # output view of y, with consumers
  z  <- ggml_scale(g, v, 2)
  z2 <- ggml_scale(g, v, 3)
  u  <- ggml_add(g, z, z2)              # z, z2 are outputs -> u needs new memory
  for (t in list(v, z, z2, u)) ggml_set_output(t)
  gr <- ggml_build_forward_expand(g, u)
  ggml_graph_expand(gr, v)

  s <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit(ggml_backend_sched_free(s), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(s, gr)
  ggml_backend_sched_graph_compute(s, gr)
  list(v = ggml_backend_tensor_get_data(v), u = ggml_backend_tensor_get_data(u))
}

test_that("an output view keeps its memory (CPU)", {
  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be), add = TRUE, after = FALSE)
  r <- alloc_view_out(be)
  expect_equal(r$v, 2 * (1:8))
  expect_equal(r$u, 10 * (1:8))
})

test_that("an output view keeps its memory (Vulkan)", {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
  be <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(be), add = TRUE, after = FALSE)
  r <- alloc_view_out(be)
  expect_equal(r$v, 2 * (1:8))
  expect_equal(r$u, 10 * (1:8))
})
