# GGML_OP_OUT_PROD on the GPU (ggmlR extension -- no upstream Vulkan shader).
#
# out_prod is what BOTH gradients of ggml_mul_mat() are built from, so before
# this shader existed every training step of every dense layer left the GPU for
# it. These compare the Vulkan result against the CPU one across the layouts the
# backward pass actually produces:
#
#   * plain contiguous inputs
#   * a TRANSPOSED src1 -- the second mul_mat gradient passes
#     ggml_transpose(grad), so this is not an exotic case but half of all uses
#   * broadcasting over dims 2/3, as grouped-query attention does
#
# dst[i0,i1,i2,i3] = sum_{i01} src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

skip_no_gpu_op <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

# Run one out_prod on the requested backend and return the result.
# `transpose_b` mirrors what ggml_build_backward_expand() does for the second
# mul_mat gradient.
run_out_prod <- function(use_gpu, a_data, b_data,
                         ne00, ne01, ne02, ne03,
                         ne10, ne12, ne13,
                         transpose_b = FALSE) {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne00, ne01, ne02, ne03)
  b <- if (transpose_b) {
    # Build it in the transposed shape and flip it, so the op sees a
    # non-contiguous src1 exactly as the backward pass hands it one.
    bt <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne01, ne10, ne12, ne13)
    ggml_set_input(bt)
    bt
  } else {
    ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne10, ne01, ne12, ne13)
  }
  ggml_set_input(a)
  if (!transpose_b) ggml_set_input(b)

  b_op <- if (transpose_b) ggml_transpose(ctx, b) else b
  out  <- ggml_out_prod(ctx, a, b_op)

  backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(a, a_data)
  ggml_backend_tensor_set_data(b, b_data)

  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, gf)
  ggml_backend_tensor_get_data(out)
}

check_out_prod <- function(ne00, ne01, ne02, ne03, ne10, ne12, ne13,
                           transpose_b = FALSE, tol = 1e-3, seed = 42) {
  set.seed(seed)
  a_data <- rnorm(ne00 * ne01 * ne02 * ne03)
  b_len  <- if (transpose_b) ne01 * ne10 * ne12 * ne13 else ne10 * ne01 * ne12 * ne13
  b_data <- rnorm(b_len)

  cpu <- run_out_prod(FALSE, a_data, b_data, ne00, ne01, ne02, ne03,
                      ne10, ne12, ne13, transpose_b)
  gpu <- run_out_prod(TRUE,  a_data, b_data, ne00, ne01, ne02, ne03,
                      ne10, ne12, ne13, transpose_b)

  expect_equal(length(gpu), length(cpu))
  expect_lt(max(abs(cpu - gpu)), tol)
  # A shader that wrote nothing would also "match" an all-zero CPU result, so
  # check the values are not degenerate.
  expect_gt(max(abs(cpu)), 1e-6)
}

test_that("out_prod matches between CPU and GPU, contiguous", {
  skip_on_cran()
  skip_no_gpu_op()
  check_out_prod(ne00 = 8L, ne01 = 5L, ne02 = 1L, ne03 = 1L,
                 ne10 = 4L, ne12 = 1L, ne13 = 1L)
})

test_that("out_prod matches on a larger contiguous case", {
  skip_on_cran()
  skip_no_gpu_op()
  check_out_prod(ne00 = 64L, ne01 = 33L, ne02 = 1L, ne03 = 1L,
                 ne10 = 17L, ne12 = 1L, ne13 = 1L, seed = 7)
})

test_that("out_prod matches with a transposed src1", {
  # The layout the second mul_mat gradient produces -- ggml_transpose(grad).
  skip_on_cran()
  skip_no_gpu_op()
  check_out_prod(ne00 = 8L, ne01 = 5L, ne02 = 1L, ne03 = 1L,
                 ne10 = 4L, ne12 = 1L, ne13 = 1L, transpose_b = TRUE)
})

test_that("out_prod matches across dims 2 and 3", {
  skip_on_cran()
  skip_no_gpu_op()
  check_out_prod(ne00 = 6L, ne01 = 4L, ne02 = 3L, ne03 = 2L,
                 ne10 = 5L, ne12 = 3L, ne13 = 2L, seed = 11)
})

test_that("out_prod matches when dims 2/3 broadcast", {
  # Grouped-query attention: several dst slices share one src0 slice.
  skip_on_cran()
  skip_no_gpu_op()
  check_out_prod(ne00 = 6L, ne01 = 4L, ne02 = 1L, ne03 = 1L,
                 ne10 = 5L, ne12 = 2L, ne13 = 2L, seed = 13)
})

test_that("a dense layer trains identically on CPU and GPU", {
  # The end-to-end reason this shader exists: mul_mat's gradients go through
  # out_prod, so a plain dense model exercises it on every step. If the shader
  # were wrong, the two backends would diverge here even though each looks
  # self-consistent.
  skip_on_cran()
  skip_no_gpu_op()

  set.seed(5)
  n <- 128L
  x <- matrix(runif(n * 4L, -1, 1), nrow = n)
  y <- matrix(rowSums(x[, 1:2]) - x[, 3], ncol = 1L)

  fit_on <- function(be) {
    set.seed(1L)
    inp <- ggml_input(shape = 4L)
    out <- inp |> ggml_layer_dense(8L, activation = "relu") |>
      ggml_layer_dense(1L)
    m <- ggml_model(inputs = inp, outputs = out)
    m <- ggml_compile(m, optimizer = "adam", loss = "mse", backend = be)
    m <- ggml_fit(m, x, y, epochs = 15L, batch_size = 32L, verbose = 0L)
    tail(m$history$train_loss, 1L)
  }

  cpu <- fit_on("cpu")
  gpu <- fit_on("vulkan")

  expect_true(is.finite(cpu) && is.finite(gpu))
  expect_equal(gpu, cpu, tolerance = 1e-2)
})

# ---------------------------------------------------------------------------
# GGML_OP_CROSS_ENTROPY_LOSS_BACK on the GPU (ggmlR extension)
#
# The other op that used to leave the GPU on every training step, this one for
# any classification model:
#     dst[row] = (softmax(logits[row]) - labels[row]) * grad[0] / nrows
# ---------------------------------------------------------------------------

run_ce_back <- function(use_gpu, logits_v, labels_v, grad_v, nc, nr) {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  g   <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1L)
  s0f <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr)
  s1f <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr)
  for (t in list(g, s0f, s1f)) ggml_set_input(t)

  out <- ggml_cross_entropy_loss_back(ctx, g, s0f, s1f)

  backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  if (!use_gpu) ggml_backend_cpu_set_n_threads(backend, 2L)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  ggml_backend_tensor_set_data(g,   grad_v)
  ggml_backend_tensor_set_data(s0f, logits_v)
  ggml_backend_tensor_set_data(s1f, labels_v)

  ggml_backend_graph_compute(backend, ggml_build_forward_expand(ctx, out))
  ggml_backend_tensor_get_data(out)
}

# The closed form, computed in R -- both backends are checked against it, not
# only against each other.
ce_back_reference <- function(logits_v, labels_v, grad_v, nc, nr) {
  lg <- matrix(logits_v, nc, nr)
  lb <- matrix(labels_v, nc, nr)
  out <- vapply(seq_len(nr), function(j) {
    z  <- lg[, j] - max(lg[, j])
    sm <- exp(z) / sum(exp(z))
    (sm - lb[, j]) * grad_v / nr
  }, numeric(nc))
  as.vector(out)
}

check_ce_back <- function(nc, nr, seed = 42, tol = 1e-4) {
  set.seed(seed)
  logits_v <- rnorm(nc * nr, sd = 2)          # wide logits stress the max-shift
  labels_v <- as.vector(vapply(seq_len(nr), function(j) {
    v <- numeric(nc); v[sample(nc, 1L)] <- 1; v
  }, numeric(nc)))
  grad_v <- 1.5

  ref <- ce_back_reference(logits_v, labels_v, grad_v, nc, nr)
  cpu <- run_ce_back(FALSE, logits_v, labels_v, grad_v, nc, nr)
  gpu <- run_ce_back(TRUE,  logits_v, labels_v, grad_v, nc, nr)

  expect_equal(cpu, ref, tolerance = tol)
  expect_lt(max(abs(gpu - cpu)), tol)
}

test_that("cross_entropy_loss_back matches CPU and the closed form", {
  skip_on_cran()
  skip_no_gpu_op()
  check_ce_back(nc = 10L, nr = 8L)
})

test_that("cross_entropy_loss_back handles a row longer than the block", {
  # BLOCK_SIZE is 32, so this forces the strided loop and the shared-memory
  # reduction to run more than one pass over a row.
  skip_on_cran()
  skip_no_gpu_op()
  check_ce_back(nc = 200L, nr = 5L, seed = 3)
})

test_that("cross_entropy_loss_back survives large logits", {
  # exp() would overflow without subtracting the row max; this is the case an
  # untrained model produces.
  skip_on_cran()
  skip_no_gpu_op()
  set.seed(9)
  nc <- 16L; nr <- 4L
  logits_v <- rnorm(nc * nr, mean = 90, sd = 5)
  labels_v <- as.vector(vapply(seq_len(nr), function(j) {
    v <- numeric(nc); v[sample(nc, 1L)] <- 1; v
  }, numeric(nc)))

  ref <- ce_back_reference(logits_v, labels_v, 1.0, nc, nr)
  gpu <- run_ce_back(TRUE, logits_v, labels_v, 1.0, nc, nr)

  expect_true(all(is.finite(gpu)))
  expect_equal(gpu, ref, tolerance = 1e-4)
})

test_that("a classifier trains identically on CPU and GPU", {
  # End to end: cross-entropy training drives this op every step, so a wrong
  # shader shows up as the two backends diverging.
  skip_on_cran()
  skip_no_gpu_op()

  set.seed(5)
  n <- 128L
  x <- matrix(runif(n * 4L, -1, 1), nrow = n)
  cls <- as.integer(rowSums(x) > 0)
  y <- cbind(1 - cls, cls) * 1.0

  fit_on <- function(be) {
    set.seed(1L)
    inp <- ggml_input(shape = 4L)
    out <- inp |> ggml_layer_dense(8L, activation = "relu") |>
      ggml_layer_dense(2L, activation = "softmax")
    m <- ggml_model(inputs = inp, outputs = out)
    m <- ggml_compile(m, optimizer = "adam",
                      loss = "categorical_crossentropy", backend = be)
    m <- ggml_fit(m, x, y, epochs = 15L, batch_size = 32L, verbose = 0L)
    tail(m$history$train_loss, 1L)
  }

  cpu <- fit_on("cpu")
  gpu <- fit_on("vulkan")
  expect_true(is.finite(cpu) && is.finite(gpu))
  expect_equal(gpu, cpu, tolerance = 1e-2)
})
