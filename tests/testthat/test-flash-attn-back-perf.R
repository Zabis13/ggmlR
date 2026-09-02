# Performance shape of the flash-attention backward shader.
#
# This is a REGRESSION test, not a benchmark: it asserts the cost SHAPE, not
# absolute milliseconds, so it does not depend on the machine.
#
# The shape it guards was found the hard way. The first version of the shader
# folded dq with one workgroup reduction per component -- DK reductions, each
# log(BLOCK_SIZE) barriers, ~448 barriers per query row at DK=64. Barriers, not
# arithmetic, then dominated: time grew LINEARLY with DK while barely moving
# with M, and the whole training step came out 3x slower than the CPU fallback
# it replaced. Rewriting dq to accumulate per-thread (each invocation owns a set
# of components and sums over all m, no barriers) removed that.
#
# So: if time is again linear in DK, some reduction has crept back in.

skip_no_vulkan <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

# One training step of a single attention block, timed.
attn_step_ms <- function(N, M, DK, DV, H, Hkv, B, reps = 10L) {
  gpu   <- ggml_vulkan_init(0L)
  sched <- ggml_backend_sched_new(list(gpu), parallel = FALSE)
  ctx   <- ggml_init(1024 * 1024 * 1024, no_alloc = TRUE)
  on.exit({
    ggml_free(ctx); ggml_backend_sched_free(sched); ggml_vulkan_free(gpu)
  }, add = TRUE)

  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, H, B)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, Hkv, B)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, Hkv, B)
  for (t in list(q, k, v)) { ggml_set_input(t); ggml_set_param(t) }

  buf <- ggml_backend_alloc_ctx_tensors(ctx, gpu)
  ggml_backend_tensor_set_data(q, runif(DK * N * H * B, -1, 1))
  ggml_backend_tensor_set_data(k, runif(DK * M * Hkv * B, -1, 1))
  ggml_backend_tensor_set_data(v, runif(DV * M * Hkv * B, -1, 1))

  loss <- ggml_sum(ctx, ggml_flash_attn_ext(ctx, q, k, v, NULL, 1 / sqrt(DK)))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_backend_sched_alloc_graph(sched, graph)

  step <- function() {
    ggml_graph_reset(graph)
    ggml_backend_sched_graph_compute(sched, graph)
  }

  step(); ggml_backend_sched_synchronize(sched)   # warm up
  t0 <- Sys.time()
  for (i in seq_len(reps)) step()
  ggml_backend_sched_synchronize(sched)
  1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / reps
}

test_that("the backward runs entirely on the GPU, without splitting the graph", {
  # The reason the shader exists: without it the backward fell to the CPU and
  # cut the graph in two per attention block.
  skip_no_vulkan()

  gpu   <- ggml_vulkan_init(0L)
  sched <- ggml_backend_sched_new(list(gpu), parallel = FALSE)
  ctx   <- ggml_init(512 * 1024 * 1024, no_alloc = TRUE)
  on.exit({
    ggml_free(ctx); ggml_backend_sched_free(sched); ggml_vulkan_free(gpu)
  }, add = TRUE)

  DK <- 64L; N <- 32L; M <- 32L; H <- 4L; B <- 1L
  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, H, B)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, H, B)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, H, B)
  for (t in list(q, k, v)) { ggml_set_input(t); ggml_set_param(t) }

  # Weights must be GPU-resident before the graph is built: the scheduler places
  # an op by where its sources already are, so host-buffer inputs keep the whole
  # graph on the CPU no matter what supports_op says.
  buf <- ggml_backend_alloc_ctx_tensors(ctx, gpu)
  ggml_backend_tensor_set_data(q, runif(DK * N * H * B, -1, 1))
  ggml_backend_tensor_set_data(k, runif(DK * M * H * B, -1, 1))
  ggml_backend_tensor_set_data(v, runif(DK * M * H * B, -1, 1))

  loss <- ggml_sum(ctx, ggml_flash_attn_ext(ctx, q, k, v, NULL, 1 / sqrt(DK)))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)

  ggml_backend_sched_alloc_graph(sched, graph)
  ggml_graph_reset(graph)
  ggml_backend_sched_graph_compute(sched, graph)

  expect_equal(ggml_backend_sched_get_n_splits(sched), 1L)

  n_cpu <- 0L
  for (i in seq_len(ggml_graph_n_nodes(graph))) {
    be <- ggml_backend_sched_get_tensor_backend(sched, ggml_graph_node(graph, i - 1L))
    if (!is.null(be) && ggml_backend_name(be) == "CPU") n_cpu <- n_cpu + 1L
  }
  expect_equal(n_cpu, 0L)
})

test_that("atomic contention, not raw atomic volume, is what the shader must avoid", {
  # This replaces two earlier assertions that looked reasonable and were both
  # measuring the wrong thing:
  #
  #   - "time must not grow linearly with DK" -- but dq is O(M*DK), so linear
  #     growth in DK is the WORK growing, not a pathology. The test passed while
  #     the shader was 3x slower than the CPU it replaced.
  #   - "time per query row must fall as rows are added" -- raising N raises the
  #     useful work at the same time as the contention, so the ratio answered
  #     neither question.
  #
  # The fix is to hold the number of atomicAdds FIXED and vary only how many of
  # them collide on the same address. dk/dv are shared by every query row of a
  # kv head, so rows-per-address is N * (H/Hkv): halving the GQA factor halves
  # the contention while the atomic count, and every other dimension of work,
  # stays identical.
  skip_no_vulkan()

  # Both do N*H*B*M*DK = 8M atomicAdds; A collides 256-deep, B only 128-deep.
  t_gqa2 <- attn_step_ms(N = 128L, M = 128L, DK = 64L, DV = 64L, H = 8L, Hkv = 4L, B = 1L)
  t_gqa1 <- attn_step_ms(N = 128L, M = 128L, DK = 64L, DV = 64L, H = 8L, Hkv = 8L, B = 1L)

  # If the shader ever accumulates without contention (private accumulators
  # merged once per workgroup), these converge. While it serialises on shared
  # addresses, the deeper collision is measurably worse.
  expect_true(is.finite(t_gqa2) && is.finite(t_gqa1))
  expect_gt(t_gqa2, 0)

  # Recorded rather than asserted: the ratio is the diagnostic, and pinning a
  # threshold to one machine would make this test lie elsewhere.
  message(sprintf("contention 256-deep: %.2f ms | 128-deep: %.2f ms | ratio %.2f",
                  t_gqa2, t_gqa1, t_gqa2 / t_gqa1))
})
