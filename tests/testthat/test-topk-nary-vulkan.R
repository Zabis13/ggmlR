# top_k on Vulkan: topk_nary_search.comp, synced from ggml 0.27.
#
# The 0.11 shader searched the float bit patterns as integers. That orders
# non-negative floats correctly and negative ones backwards, so a row with
# negative values got the wrong top elements; a NaN had no place in the search
# range at all, and the end of the top bucket could wrap past 2^32. 0.27 ranks
# NaN like -inf, compares as floats, and clamps the range.
#
# The n-ary search is only used for wide rows on devices with full subgroup
# support (otherwise top_k falls back to the argsort shader), so the rows here
# are wide; on a device without that support these tests still pass, they just
# exercise the fallback. Values are distinct, so ties play no part -- those are
# covered by test-argsort-ties-vulkan.R.
#
# ggml_top_k promises the top k indices "in no particular order" (ggml.h):
# the CPU happens to return them by value, the n-ary search by index. So the
# checks compare the sets, sorted, not the sequences.

run_top_k <- function(values, k, use_gpu) {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(values))
  out <- ggml_top_k(ctx, a, k)
  gf <- ggml_build_forward_expand(ctx, out)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, values)
  ggml_backend_graph_compute(backend, gf)
  as.integer(ggml_backend_tensor_get_data(out))
}

# 0-based indices of the k largest values; NaN never selected.
expected_top_k <- function(values, k) {
  ok <- which(!is.na(values))
  ok[order(values[ok], decreasing = TRUE)][seq_len(k)] - 1L
}

# Distinct values that survive the round trip to float32.
distinct_f32 <- function(n, lo, hi) {
  v <- unique(round(runif(n * 2, lo, hi), 3))
  v[seq_len(n)]
}

test_that("Vulkan top_k picks the largest of all-negative values", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  set.seed(11L)
  v <- distinct_f32(4096, -1000, -1)
  for (k in c(1L, 8L, 64L)) {
    gpu <- run_top_k(v, k, use_gpu = TRUE)
    expect_equal(sort(gpu), sort(expected_top_k(v, k)), label = sprintf("k = %d", k))
    expect_equal(sort(gpu), sort(run_top_k(v, k, use_gpu = FALSE)))
  }
})

test_that("Vulkan top_k picks the largest of mixed-sign values", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  set.seed(12L)
  v <- distinct_f32(4096, -500, 500)
  for (k in c(8L, 256L)) {
    gpu <- run_top_k(v, k, use_gpu = TRUE)
    expect_equal(sort(gpu), sort(expected_top_k(v, k)), label = sprintf("k = %d", k))
    expect_equal(sort(gpu), sort(run_top_k(v, k, use_gpu = FALSE)))
  }
})

test_that("Vulkan top_k never selects NaN", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  set.seed(13L)
  v <- distinct_f32(4096, -100, 100)
  v[sample.int(length(v), 64)] <- NaN
  for (k in c(8L, 64L)) {
    gpu <- run_top_k(v, k, use_gpu = TRUE)
    expect_false(any(is.nan(v[gpu + 1L])), label = sprintf("NaN picked, k = %d", k))
    expect_equal(sort(gpu), sort(expected_top_k(v, k)), label = sprintf("k = %d", k))
  }
})

test_that("Vulkan top_k handles values at the top of the float range", {
  # Large positive floats have the highest bit patterns, where the 0.11 search
  # range ended (0xFF800000) and the top bucket's end wrapped past 2^32.
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  set.seed(14L)
  v <- c(distinct_f32(4092, -1, 1), 3e38, 2e38, 1e38, Inf)
  v <- v[sample.int(length(v))]
  gpu <- run_top_k(v, 4L, use_gpu = TRUE)
  expect_equal(sort(gpu), sort(expected_top_k(v, 4L)))
})
