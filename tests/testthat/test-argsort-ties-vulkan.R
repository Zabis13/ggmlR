# argsort / top_k with tied values.
#
# The comparators used to test only "a < b", which leaves equal elements in
# whatever order the sort happened to move them: std::sort is not stable, and
# the bitonic network in the shaders is not either.  That is invisible while
# every value is distinct and decisive when they are not -- MaskRCNN quantises
# its RPN scores to uint8, so 594 proposals share 118 distinct values and one
# tie group holds 69 of them.  Which of those survive TopK decides the whole
# detection set, and the model returned 48 boxes where onnxruntime returns 51.
#
# Both comparators now break ties by index, lower first.  These tests pin that
# down on purpose-built ties and check the CPU and Vulkan paths agree, since
# they implement it separately -- and in argsort.comp the tie direction has to
# follow p.order, because DESC there is served by reading an ascending sort
# backwards, which reverses ties along with everything else.

run_sort <- function(values, op, order = NULL, k = NULL, use_gpu = FALSE) {
  ctx <- ggml_init(16 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx))

  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(values))
  out <- if (op == "argsort") ggml_argsort(ctx, a, order) else ggml_top_k(ctx, a, k)
  gf <- ggml_build_forward_expand(ctx, out)

  backend <- if (use_gpu) ggml_vulkan_init(0) else ggml_backend_cpu_init()
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, values)
  ggml_backend_graph_compute(backend, gf)
  # as.integer: these are indices, and whether the binding hands them back as
  # integer or double is not what the test is about.
  as.integer(ggml_backend_tensor_get_data(out))
}

# Four distinct values over sixteen slots: every value is a tie group of four,
# so the index order is the only thing that can decide the result.
tied_values <- rep(c(3, 1, 4, 1), each = 4)

test_that("argsort ascending orders ties by index", {
  got <- run_sort(tied_values, "argsort", order = GGML_SORT_ORDER_ASC)
  expect_equal(got, order(tied_values, seq_along(tied_values)) - 1L)
})

test_that("argsort descending orders ties by index", {
  got <- run_sort(tied_values, "argsort", order = GGML_SORT_ORDER_DESC)
  expect_equal(got, order(-tied_values, seq_along(tied_values)) - 1L)
})

test_that("top_k returns the top elements in descending order", {
  # Sorted, not set-compared: the kernel used to swap its first two entries
  # "to emphasize that the order is not important", which ONNX TopK -- whose
  # sorted=1 promises an order -- was reading as the ranking.
  v <- c(1, 6, 2, 5, 3, 4)
  got <- run_sort(v, "top_k", k = 3)
  expect_equal(got, c(1L, 3L, 5L))
})

test_that("top_k orders ties by index", {
  got <- run_sort(tied_values, "top_k", k = 8)
  expect_equal(got, (order(-tied_values, seq_along(tied_values)) - 1L)[1:8])
})

test_that("argsort is deterministic across repeated runs", {
  # A tie order that comes from the sort's internal state can differ between
  # runs of the same binary; one that comes from the comparator cannot.
  set.seed(1L)
  v <- as.numeric(sample.int(8, 256, replace = TRUE))
  first <- run_sort(v, "argsort", order = GGML_SORT_ORDER_DESC)
  for (i in 1:3)
    expect_equal(run_sort(v, "argsort", order = GGML_SORT_ORDER_DESC), first)
})

test_that("Vulkan agrees with the CPU on tied argsort", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  set.seed(2L)
  # Values drawn from a small set, so ties are the rule rather than the
  # exception -- 256 elements over 8 distinct values.
  v <- as.numeric(sample.int(8, 256, replace = TRUE))
  for (ord in c(GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC)) {
    cpu <- run_sort(v, "argsort", order = ord, use_gpu = FALSE)
    gpu <- run_sort(v, "argsort", order = ord, use_gpu = TRUE)
    expect_equal(gpu, cpu)
  }
})

test_that("Vulkan agrees with the CPU on tied top_k", {
  skip_if_not(ggml_vulkan_available(), "no Vulkan device")
  set.seed(3L)
  v <- as.numeric(sample.int(8, 256, replace = TRUE))
  for (k in c(4L, 32L)) {
    cpu <- run_sort(v, "top_k", k = k, use_gpu = FALSE)
    gpu <- run_sort(v, "top_k", k = k, use_gpu = TRUE)
    expect_equal(gpu, cpu)
  }
})
