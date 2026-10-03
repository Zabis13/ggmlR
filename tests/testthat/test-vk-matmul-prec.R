# GGML_PREC_F32 on Vulkan f32 x f32 matmul.
#
# On fp16/coopmat devices the mat-mat kernel (n > 8; n <= 8 is the mat-vec
# path, always f32) keeps its tiles in f16: ~4e-4 relative error. With the
# flag the backend must pick the f32-tile pipeline -- checked by the numbers
# AND by a dispatch counter, so a later change of the selection cannot leave
# the test green by accident (on an fp32-only device the numbers would pass
# either way). Transposed (non-contiguous) and batched operands must stay f32
# too: that is what a matmul backward feeds the kernel.

skip_if_no_vk <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}

PREC_F32 <- 10L
prec32_count <- function() .Call("R_ggml_vk_prec32_count", PACKAGE = "ggmlR")
set_prec32   <- function(node) .Call("R_ggml_mul_mat_set_prec", node, PREC_F32, PACKAGE = "ggmlR")

# Build tensors from `inputs` (list of list(ne =, data =)), run build(g, ts) on
# `be`, return the output as a numeric vector.
run_graph <- function(be, inputs, build) {
  cs <- ggml_init(512e6, no_alloc = TRUE)
  on.exit(ggml_free(cs), add = TRUE, after = FALSE)
  ts <- lapply(inputs, function(i) switch(length(i$ne),
    ggml_new_tensor_1d(cs, GGML_TYPE_F32, i$ne[1]),
    ggml_new_tensor_2d(cs, GGML_TYPE_F32, i$ne[1], i$ne[2]),
    ggml_new_tensor_3d(cs, GGML_TYPE_F32, i$ne[1], i$ne[2], i$ne[3])))
  buf <- ggml_backend_alloc_ctx_tensors(cs, be)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  for (nm in names(inputs)) ggml_backend_tensor_set_data(ts[[nm]], inputs[[nm]]$data)
  g <- ggml_init(64e6, no_alloc = TRUE)
  on.exit(ggml_free(g), add = TRUE, after = FALSE)
  out <- build(g, ts)
  ggml_set_output(out)
  gr <- ggml_build_forward_expand(g, out)
  s <- ggml_backend_sched_new(list(be), parallel = FALSE)
  on.exit(ggml_backend_sched_free(s), add = TRUE, after = FALSE)
  ggml_backend_sched_alloc_graph(s, gr)
  ggml_backend_sched_graph_compute(s, gr)
  ggml_backend_tensor_get_data(out)
}

# W %*% X through ggml_mul_mat. tB: pass X as a transposed VIEW (non-contiguous)
# instead of a contiguous tensor. Only src1 can be: ggml_mul_mat asserts that
# src0 is not transposed, so a backward graph never feeds the kernel one.
mm <- function(be, W, X, prec, tB = FALSE) {
  m <- nrow(W); k <- ncol(W); n <- ncol(X)
  inputs <- list(
    a = list(ne = c(k, m), data = as.numeric(t(W))),
    b = if (tB) list(ne = c(n, k), data = as.numeric(t(X))) else list(ne = c(k, n), data = as.numeric(X)))
  out <- run_graph(be, inputs, function(g, ts) {
    a <- ts$a
    b <- if (tB) ggml_transpose(g, ts$b) else ts$b
    r <- ggml_mul_mat(g, a, b)
    if (prec) set_prec32(r)
    r
  })
  matrix(out, m, n)
}

rel_err <- function(x, ref) max(abs(x - ref)) / max(abs(ref))

with_vk <- function(code) {
  be <- ggml_vulkan_init(0L)
  on.exit(ggml_backend_free(be), add = TRUE)
  code(be)
}

test_that("GGML_PREC_F32 gives f32 accuracy across the mat-vec / mat-mat border", {
  skip_if_no_vk()
  with_vk(function(be) {
    set.seed(11)
    W <- matrix(rnorm(64 * 9, sd = 0.5), 64, 9)
    for (n in c(1L, 8L, 9L, 16L, 128L, 2048L)) {
      X <- matrix(rnorm(9 * n), 9, n)
      ref <- W %*% X
      c0 <- prec32_count()
      got <- mm(be, W, X, prec = TRUE)
      expect_lt(rel_err(got, ref), 1e-5)
      # mat-vec (n <= 8) never reaches the mat-mat dispatcher
      if (n <= 8) expect_identical(prec32_count(), c0, info = paste("n =", n))
      else        expect_gt(prec32_count(), c0)
      # without the flag: unchanged behaviour, and never the prec32 path
      c1 <- prec32_count()
      mm(be, W, X, prec = FALSE)
      expect_identical(prec32_count(), c1, info = paste("n =", n))
    }
  })
})

test_that("GGML_PREC_F32 holds on shapes off the tile grid and on every tile size", {
  skip_if_no_vk()
  with_vk(function(be) {
    set.seed(12)
    # (m, k, n): s tile, m tile, l tile unaligned (k not a multiple of 128),
    # l tile aligned (k = 256)
    for (sh in list(c(37, 19, 53), c(64, 128, 40), c(300, 250, 300), c(300, 256, 300))) {
      W <- matrix(rnorm(sh[1] * sh[2]), sh[1], sh[2])
      X <- matrix(rnorm(sh[2] * sh[3]), sh[2], sh[3])
      expect_lt(rel_err(mm(be, W, X, prec = TRUE), W %*% X), 1e-5,
                label = paste(sh, collapse = "x"))
    }
  })
})

test_that("transposed operands stay f32 under GGML_PREC_F32 (matmul backward)", {
  skip_if_no_vk()
  with_vk(function(be) {
    set.seed(13)
    # forward C = W X, upstream gradient G; dX = t(W) G, dW = G t(X).
    W <- matrix(rnorm(48 * 20), 48, 20)
    X <- matrix(rnorm(20 * 96), 20, 96)
    G <- matrix(rnorm(48 * 96), 48, 96)
    c0 <- prec32_count()
    # dX: t(W) %*% G -- G passed as a transposed (non-contiguous) view
    dX <- mm(be, t(W), G, prec = TRUE, tB = TRUE)
    expect_lt(rel_err(dX, t(W) %*% G), 1e-5)
    # dW: G %*% t(X) -- t(X) passed as a transposed view of X
    dW <- mm(be, G, t(X), prec = TRUE, tB = TRUE)
    expect_lt(rel_err(dW, G %*% t(X)), 1e-5)
    expect_gte(prec32_count() - c0, 2)
    # and the CPU agrees. Contiguous operands there: the CPU mul_mat asserts
    # nb00 == sizeof(float), so a transposed view would abort R.
    cpu <- ggml_backend_cpu_init()
    on.exit(ggml_backend_free(cpu), add = TRUE)
    expect_lt(rel_err(dW, mm(cpu, G, t(X), prec = TRUE)), 1e-5)
    expect_lt(rel_err(dX, mm(cpu, t(W), G, prec = TRUE)), 1e-5)
  })
})

test_that("batched f32 matmul honours GGML_PREC_F32", {
  skip_if_no_vk()
  with_vk(function(be) {
    set.seed(14)
    m <- 24L; k <- 17L; n <- 40L; B <- 3L
    A  <- array(rnorm(k * m * B), c(k, m, B))      # ggml layout: ne0 = k, ne1 = m
    Xb <- array(rnorm(k * n * B), c(k, n, B))
    out <- run_graph(be, list(a = list(ne = c(k, m, B), data = as.numeric(A)),
                              b = list(ne = c(k, n, B), data = as.numeric(Xb))),
                     function(g, ts) { r <- ggml_mul_mat(g, ts$a, ts$b); set_prec32(r); r })
    out <- array(out, c(m, n, B))
    for (i in seq_len(B))
      expect_lt(rel_err(out[, , i], t(A[, , i]) %*% Xb[, , i]), 1e-5, label = paste("batch", i))
  })
})

test_that("prec32 and default pipelines coexist in one process", {
  skip_if_no_vk()
  with_vk(function(be) {
    set.seed(15)
    W <- matrix(rnorm(64 * 32), 64, 32)
    X <- matrix(rnorm(32 * 128), 32, 128)
    p1 <- mm(be, W, X, prec = TRUE)
    d1 <- mm(be, W, X, prec = FALSE)
    p2 <- mm(be, W, X, prec = TRUE)
    d2 <- mm(be, W, X, prec = FALSE)
    expect_identical(p2, p1)
    expect_identical(d2, d1)
    expect_lt(rel_err(p1, W %*% X), 1e-5)
  })
})

test_that("ag_matmul_precision() gets, sets and validates", {
  expect_identical(ag_matmul_precision(), "default")
  old <- ag_matmul_precision("f32")
  on.exit(ag_matmul_precision(old), add = TRUE)
  expect_identical(old, "default")
  expect_identical(ag_matmul_precision(), "f32")
  expect_error(ag_matmul_precision("f16"), "should be one of")
  expect_identical(ag_matmul_precision(), "f32")      # a bad value changes nothing
})

test_that("ag_matmul_precision('f32') reaches the forward and the graph backward", {
  skip_if_no_vk()
  ag_device("gpu")
  old <- ag_matmul_precision("f32")
  on.exit({ ag_matmul_precision(old); ag_device("cpu") }, add = TRUE)

  set.seed(16)
  W0 <- matrix(rnorm(64 * 9, sd = 0.5), 64, 9)
  X0 <- matrix(rnorm(9 * 128), 9, 128)
  Y0 <- matrix(rnorm(64 * 128), 64, 128)
  W <- ag_param(W0); X <- ag_param(X0)

  c0 <- prec32_count()
  with_grad_tape({ out <- ag_matmul(W, X); loss <- ag_mse_loss(out, ag_tensor(Y0)) })
  expect_lt(rel_err(as.matrix(out), W0 %*% X0), 1e-5)
  c1 <- prec32_count()
  expect_gt(c1, c0)                                   # forward took the f32 path

  g <- backward(loss)
  expect_identical(ggmlR:::ag_backward_path(), "graph")
  expect_gt(prec32_count(), c1)                       # so did dX in the backward
  G  <- 2 * (W0 %*% X0 - Y0) / length(Y0)
  gx <- ggmlR:::.ag_as_matrix(get0(as.character(X$id), envir = g))
  gw <- ggmlR:::.ag_as_matrix(get0(as.character(W$id), envir = g))
  expect_lt(rel_err(gx, t(W0) %*% G), 1e-5)           # mul_mat, flagged
  expect_lt(rel_err(gw, G %*% t(X0)), 1e-5)           # out_prod, f32 shader
})
