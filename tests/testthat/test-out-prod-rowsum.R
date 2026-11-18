# GGML_OP_OUT_PROD_ROWSUM (ggmlR extension): ggml_out_prod(a, b) with one more
# column, the row sums of a -- the weight and bias gradients of a dense layer in
# one pass over the output gradient.
#
#   dst[i0, i1]  = sum_k a[i0, k] * b[i1, k]    i1 < ne10
#   dst[i0, ne10] = sum_k a[i0, k]
#
# The CPU kernel is the reference the Vulkan shader is checked against, so it
# is checked here against plain R arithmetic, against ggml_out_prod on the
# shared columns, and for bit-identical results across runs and thread counts.

# One op on the CPU backend; `b_data` in b's own (possibly transposed) layout.
run_rowsum <- function(a_data, b_data, ne00, ne01, ne02, ne03, ne10, ne12, ne13,
                       transpose_b = FALSE, threads = 2L, op = "rowsum", gpu = FALSE) {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne00, ne01, ne02, ne03)
  b <- if (transpose_b) ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne01, ne10, ne12, ne13)
       else ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne10, ne01, ne12, ne13)
  ggml_set_input(a); ggml_set_input(b)
  b_op <- if (transpose_b) ggml_transpose(ctx, b) else b
  out <- if (op == "rowsum") ggml_out_prod_rowsum(ctx, a, b_op) else ggml_out_prod(ctx, a, b_op)
  backend <- if (gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  if (!gpu) ggml_backend_cpu_set_n_threads(backend, threads)
  if (gpu) stopifnot(ggml_backend_dev_supports_op(ggml_backend_get_device(backend), out))
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, a_data)
  ggml_backend_tensor_set_data(b, b_data)
  ggml_backend_graph_compute(backend, ggml_build_forward_expand(ctx, out))
  ggml_backend_tensor_get_data(out)
}

# The same in R, double precision: dst [ne00, ne10 + 1, ne12, ne13].
ref_rowsum <- function(a_data, b_data, ne00, ne01, ne02, ne03, ne10, ne12, ne13,
                       transpose_b = FALSE) {
  A <- array(a_data, c(ne00, ne01, ne02, ne03))
  B <- if (transpose_b) aperm(array(b_data, c(ne01, ne10, ne12, ne13)), c(2, 1, 3, 4))
       else array(b_data, c(ne10, ne01, ne12, ne13))
  out <- array(0, c(ne00, ne10 + 1, ne12, ne13))
  for (i3 in seq_len(ne13)) for (i2 in seq_len(ne12)) {
    a2 <- matrix(A[, , (i2 - 1) %/% (ne12 / ne02) + 1, (i3 - 1) %/% (ne13 / ne03) + 1], ne00, ne01)
    b2 <- matrix(B[, , i2, i3], ne10, ne01)
    out[, , i2, i3] <- cbind(a2 %*% t(b2), rowSums(a2))
  }
  as.vector(out)
}

check_rowsum <- function(ne00, ne01, ne02, ne03, ne10, ne12, ne13,
                         transpose_b = FALSE, seed = 1) {
  set.seed(seed)
  a <- rnorm(ne00 * ne01 * ne02 * ne03, mean = 0.3)   # mean != 0: row sums not ~0
  b <- rnorm(ne10 * ne01 * ne12 * ne13)
  got <- run_rowsum(a, b, ne00, ne01, ne02, ne03, ne10, ne12, ne13, transpose_b)
  ref <- ref_rowsum(a, b, ne00, ne01, ne02, ne03, ne10, ne12, ne13, transpose_b)
  expect_length(got, ne00 * (ne10 + 1) * ne12 * ne13)
  # f32 accumulation over ne01 terms against double
  expect_lt(max(abs(got - ref)) / max(abs(ref)), 1e-5 * max(1, sqrt(ne01) / 8))
  # the first ne10 columns are ggml_out_prod's
  op <- run_rowsum(a, b, ne00, ne01, ne02, ne03, ne10, ne12, ne13, transpose_b, op = "out_prod")
  cols <- array(got, c(ne00, ne10 + 1, ne12, ne13))[, seq_len(ne10), , , drop = FALSE]
  expect_lt(max(abs(as.vector(cols) - op)) / max(abs(op)), 1e-6)
}

test_that("out_prod_rowsum: dense-layer shapes match R", {
  check_rowsum(64, 2048, 1, 1, 9, 1, 1)      # PPO first layer: g [64, B], x [9, B]
  check_rowsum(64, 2048, 1, 1, 64, 1, 1)
  check_rowsum(256, 256, 1, 1, 64, 1, 1)
  check_rowsum(9, 512, 1, 1, 64, 1, 1)       # narrow output (logits)
})

test_that("out_prod_rowsum: edges -- k = 1, n = 1, sizes off any tile", {
  check_rowsum(5, 1, 1, 1, 3, 1, 1)
  check_rowsum(33, 7, 1, 1, 1, 1, 1)          # dst 2 columns: one product, one sum
  check_rowsum(1, 100, 1, 1, 31, 1, 1)
  check_rowsum(37, 65, 1, 1, 33, 1, 1)
})

test_that("out_prod_rowsum: transposed b, as the backward pass hands it", {
  check_rowsum(64, 300, 1, 1, 9, 1, 1, transpose_b = TRUE)
  check_rowsum(33, 70, 1, 1, 40, 1, 1, transpose_b = TRUE)
})

test_that("out_prod_rowsum: dims 2 and 3, with and without broadcast", {
  check_rowsum(16, 40, 3, 2, 8, 3, 2)
  check_rowsum(16, 40, 1, 1, 8, 3, 2)         # one a slice for every b slice
  check_rowsum(16, 40, 3, 1, 8, 6, 1)         # a broadcast 2x over dim 2
})

test_that("out_prod_rowsum is bit-identical across runs and thread counts", {
  set.seed(7)
  a <- rnorm(64 * 2048, mean = 0.3); b <- rnorm(9 * 2048)
  r1 <- run_rowsum(a, b, 64, 2048, 1, 1, 9, 1, 1, threads = 1L)
  r2 <- run_rowsum(a, b, 64, 2048, 1, 1, 9, 1, 1, threads = 2L)
  r3 <- run_rowsum(a, b, 64, 2048, 1, 1, 9, 1, 1, threads = 2L)
  expect_identical(r1, r2)
  expect_identical(r2, r3)
})

test_that("out_prod_rowsum refuses bad inputs with an R error", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 10)
  expect_error(ggml_out_prod_rowsum(ctx, a, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 9)),
               "share ne\\[1\\]")
  expect_error(ggml_out_prod_rowsum(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 10),
                                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 10)), "F32")
  expect_error(ggml_out_prod_rowsum(ctx, ggml_transpose(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10, 4)),
                                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 10)), "transposed")
})

# The op enum lives in two copies of ggml.h (src/ for the build, inst/include/
# for packages linking libggml.a); a difference shifts op numbers between them.
test_that("the two ggml.h copies declare the same ops", {
  src <- test_path("..", "..", "src", "ggml.h")
  inc <- test_path("..", "..", "inst", "include", "ggml.h")
  skip_if_not(file.exists(src) && file.exists(inc), "source tree not available")
  ops <- function(f) {
    x <- readLines(f)
    x[seq(grep("^    enum ggml_op \\{", x), grep("^        GGML_OP_COUNT,", x))]
  }
  expect_identical(ops(src), ops(inc))
  expect_true(any(grepl("GGML_OP_OUT_PROD_ROWSUM", ops(src))))
})

# ---- Vulkan against the CPU reference -------------------------------------

skip_no_vk_rowsum <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

check_rowsum_gpu <- function(ne00, ne01, ne02, ne03, ne10, ne12, ne13,
                             transpose_b = FALSE, seed = 1) {
  set.seed(seed)
  a <- rnorm(ne00 * ne01 * ne02 * ne03, mean = 0.3)
  b <- rnorm(ne10 * ne01 * ne12 * ne13)
  cpu <- run_rowsum(a, b, ne00, ne01, ne02, ne03, ne10, ne12, ne13, transpose_b)
  gpu <- run_rowsum(a, b, ne00, ne01, ne02, ne03, ne10, ne12, ne13, transpose_b, gpu = TRUE)
  expect_length(gpu, length(cpu))
  expect_lt(max(abs(gpu - cpu)) / max(abs(cpu)), 1e-5 * max(1, sqrt(ne01) / 8))
  # the row-sum column on its own: a shader that skipped it would leave
  # whatever the buffer held
  rs <- function(v) as.vector(array(v, c(ne00, ne10 + 1, ne12 * ne13))[, ne10 + 1, ])
  expect_lt(max(abs(rs(gpu) - rs(cpu))) / max(abs(rs(cpu))), 1e-5 * max(1, sqrt(ne01) / 8))
  expect_gt(max(abs(rs(cpu))), 1e-3)
  # deterministic: a second GPU run is bit-identical
  expect_identical(run_rowsum(a, b, ne00, ne01, ne02, ne03, ne10, ne12, ne13, transpose_b, gpu = TRUE),
                   gpu)
}

test_that("out_prod_rowsum on Vulkan matches the CPU: dense-layer shapes", {
  skip_no_vk_rowsum()
  check_rowsum_gpu(64, 2048, 1, 1, 9, 1, 1)
  check_rowsum_gpu(64, 2048, 1, 1, 64, 1, 1)     # 2 x 2 tiles: sums from y == 0 only
  check_rowsum_gpu(256, 256, 1, 1, 64, 1, 1)
  check_rowsum_gpu(9, 512, 1, 1, 64, 1, 1)
  check_rowsum_gpu(64, 13312, 1, 1, 64, 1, 1)    # long reduction (cube Q batch)
})

test_that("out_prod_rowsum on Vulkan matches the CPU: tile edges", {
  skip_no_vk_rowsum()
  check_rowsum_gpu(5, 1, 1, 1, 3, 1, 1)
  check_rowsum_gpu(33, 7, 1, 1, 1, 1, 1)
  check_rowsum_gpu(1, 100, 1, 1, 31, 1, 1)
  check_rowsum_gpu(37, 65, 1, 1, 33, 1, 1)
  check_rowsum_gpu(100, 31, 1, 1, 70, 1, 1)
})

test_that("out_prod_rowsum on Vulkan matches the CPU: transposed b, dims 2 and 3", {
  skip_no_vk_rowsum()
  check_rowsum_gpu(64, 300, 1, 1, 9, 1, 1, transpose_b = TRUE)
  check_rowsum_gpu(33, 70, 1, 1, 40, 1, 1, transpose_b = TRUE)
  check_rowsum_gpu(16, 40, 3, 2, 8, 3, 2)
  check_rowsum_gpu(16, 40, 1, 1, 8, 3, 2)
  check_rowsum_gpu(16, 40, 3, 1, 8, 6, 1)
})
