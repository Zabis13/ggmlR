# GPU linear algebra drop-in: ggml_matmul / ggml_crossprod / ggml_tcrossprod and
# the as_gpu_matrix() S4 wrapper (%*% / crossprod / tcrossprod). The CPU path and
# the CPU fallback must reproduce base R exactly; the GPU path (when a Vulkan
# device is present) must match base R to f32 precision. Small multiplies and
# device = "cpu" stay on the CPU, so most assertions run without a GPU.

set.seed(42)
A <- matrix(rnorm(3 * 4), 3, 4)
B <- matrix(rnorm(4 * 5), 4, 5)
rownames(A) <- paste0("r", 1:3)
colnames(A) <- paste0("k", 1:4)
colnames(B) <- paste0("c", 1:5)

test_that("ggml_matmul on CPU reproduces %*% including dimnames", {
  out <- ggml_matmul(A, B, device = "cpu")
  expect_equal(out, A %*% B)
  expect_identical(dimnames(out), list(rownames(A), colnames(B)))
})

test_that("ggml_crossprod / ggml_tcrossprod reproduce base R on CPU", {
  expect_equal(ggml_crossprod(A, device = "cpu"), crossprod(A))
  expect_equal(ggml_crossprod(A, B[1:3, ], device = "cpu"), crossprod(A, B[1:3, ]))
  expect_equal(ggml_tcrossprod(A, device = "cpu"), tcrossprod(A))
  expect_equal(ggml_tcrossprod(A, A, device = "cpu"), tcrossprod(A, A))
})

test_that("non-conformable inputs error", {
  expect_error(ggml_matmul(A, A, device = "cpu"), "non-conformable")
  expect_error(ggml_crossprod(A, B, device = "cpu"), "non-conformable")  # 3 vs 4 rows
  expect_error(ggml_tcrossprod(A, B, device = "cpu"), "non-conformable") # 4 vs 5 cols
})

test_that("device = 'auto' below the size gate stays on the CPU (no GPU needed)", {
  # A small multiply must return the exact base-R result whether or not a GPU is
  # present, because auto keeps small problems on the CPU.
  expect_equal(ggml_matmul(A, B, device = "auto"), A %*% B)
})

test_that("as_gpu_matrix wraps and %*% dispatches to the GPU path", {
  gA <- as_gpu_matrix(A, device = "cpu")
  expect_s4_class(gA, "ggml_matrix")
  expect_identical(dim(gA), dim(A))
  expect_identical(as.matrix(gA), A)

  expect_equal(gA %*% B,               A %*% B, ignore_attr = TRUE)
  expect_equal(A %*% as_gpu_matrix(B, device = "cpu"), A %*% B, ignore_attr = TRUE)
  expect_equal(as_gpu_matrix(A, device = "cpu") %*% as_gpu_matrix(B, device = "cpu"),
               A %*% B, ignore_attr = TRUE)
})

test_that("crossprod / tcrossprod dispatch on a wrapped matrix", {
  gA <- as_gpu_matrix(A, device = "cpu")
  expect_equal(crossprod(gA),  crossprod(A),  ignore_attr = TRUE)
  expect_equal(tcrossprod(gA), tcrossprod(A), ignore_attr = TRUE)
})

test_that("a vector operand is treated as a 1-column matrix like base R", {
  v <- rnorm(4)
  expect_equal(ggml_matmul(A, v, device = "cpu"), A %*% v, ignore_attr = TRUE)
})

# ---- GPU path (only when a Vulkan device is present) -----------------------

test_that("GPU path approximates base R (driver-dependent precision)", {
  skip_if_not(isTRUE(tryCatch(
    ggml_vulkan_available() && ggml_vulkan_device_count() > 0L,
    error = function(e) FALSE)), "no Vulkan GPU")
  withr::defer(ag_device("cpu"))

  # large enough to actually exercise the GPU under device = "gpu". The GPU is a
  # fast approximate multiply: prec = "f32" requests f32 accumulation, but some
  # drivers (RADV/Mesa) accumulate mul_mat in f16 regardless (~1e-3 relative), so
  # the tolerance is loose on purpose — this is not a bit-for-bit check.
  set.seed(7)
  P <- matrix(rnorm(400 * 300), 400, 300)
  Q <- matrix(rnorm(300 * 200), 300, 200)

  relerr <- function(a, b) max(abs(a - b)) / max(abs(b))
  expect_lt(relerr(ggml_matmul(P, Q, device = "gpu", prec = "f32"),   P %*% Q),      2e-3)
  expect_lt(relerr(ggml_crossprod(P, device = "gpu", prec = "f32"),   crossprod(P)), 2e-3)
  expect_lt(relerr(ggml_tcrossprod(Q, device = "gpu", prec = "f32"),  tcrossprod(Q)),2e-3)

  # dimnames: unnamed inputs -> no dimnames on output, exactly like base R
  expect_null(dimnames(ggml_matmul(P, Q, device = "gpu")))
})

test_that("ggml_matmul_f64 on CPU is exact double, with dimnames like base R", {
  expect_equal(ggml_matmul_f64(A, B, device = "cpu"), A %*% B)
  expect_identical(dimnames(ggml_matmul_f64(A, B, device = "cpu")),
                   list(rownames(A), colnames(B)))
  expect_error(ggml_matmul_f64(A, A, device = "cpu"), "non-conformable")
  # unnamed inputs -> no dimnames, like base R
  U <- matrix(rnorm(6), 2, 3); V <- matrix(rnorm(12), 3, 4)
  expect_null(dimnames(ggml_matmul_f64(U, V, device = "cpu")))
})

test_that("ggml_matmul_f64 GPU path matches base R to double precision", {
  skip_if_not(isTRUE(tryCatch(
    ggml_vulkan_available() && ggml_vulkan_device_count() > 0L,
    error = function(e) FALSE)), "no Vulkan GPU")
  withr::defer(ag_device("cpu"))

  set.seed(11)
  P <- matrix(rnorm(200 * 150), 200, 150)
  Q <- matrix(rnorm(150 * 100), 150, 100)

  g <- ggml_matmul_f64(P, Q, device = "gpu")
  # honest double: machine precision, orders of magnitude tighter than the f32
  # path (~1e-3). On a device without fp64 support this silently falls back to the
  # CPU, which is also exact, so the assertion holds either way.
  relerr <- function(a, b) max(abs(a - b)) / max(abs(b))
  expect_lt(relerr(g, P %*% Q), 1e-10)
})
