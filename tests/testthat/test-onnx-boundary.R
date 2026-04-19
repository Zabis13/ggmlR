# test-onnx-boundary.R
#
# Boundary / regression tests targeting four bug classes:
#   1. Aliasing  — gallocr reuses buffers (repeated runs)
#   2. Tile boundary — shader reads/writes outside tile at small K/N
#   3. 4D truncation — 5D tensor copy drops last dim
#   4. Wrong op dispatch — direct conv instead of MUL_MAT 1x1 fast path
#
# Each section: unit (single op), graph (3-5 nodes), and where possible
# CPU vs GPU comparison (skips gracefully when no Vulkan device).

# ── helpers ──────────────────────────────────────────────────────────────────

run_onnx <- function(path, inputs, device = "cpu") {
  m <- onnx_load(path, device = device)
  onnx_run(m, inputs)[[1]]
}

# Run same model on CPU and GPU, return list(cpu, gpu)
run_cpu_gpu <- function(path, inputs) {
  cpu <- run_onnx(path, inputs, device = "cpu")
  gpu <- tryCatch(run_onnx(path, inputs, device = "gpu"), error = function(e) NULL)
  list(cpu = cpu, gpu = gpu)
}

# Compare CPU vs GPU with cosine + max abs tolerance
expect_cpu_gpu_close <- function(path, inputs, tol = 1e-3) {
  r <- run_cpu_gpu(path, inputs)
  skip_if(is.null(r$gpu), "No Vulkan device")
  cpu <- as.numeric(r$cpu)
  gpu <- as.numeric(r$gpu)
  expect_equal(length(cpu), length(gpu))
  max_abs <- max(abs(cpu - gpu))
  expect_true(max_abs < tol,
    label = sprintf("CPU/GPU max abs diff = %.2e (tol %.2e)", max_abs, tol))
}

# ── matmul helper with explicit K ────────────────────────────────────────────

make_matmul_k <- function(M, K, N, a_data = NULL, b_data = NULL) {
  if (is.null(a_data)) a_data <- seq_len(M * K) * 0.1
  if (is.null(b_data)) b_data <- seq_len(K * N) * 0.1
  inp_a <- .onnx_value_info("A", 1L, c(M, K))
  inp_b <- .onnx_value_info("B", 1L, c(K, N))
  outp  <- .onnx_value_info("Y", 1L, c(M, N))
  node  <- .onnx_node("MatMul", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  list(path = path, a = a_data, b = b_data)
}

# ── Conv helper ───────────────────────────────────────────────────────────────

make_conv2d <- function(N, C, H, W, OC, KH, KW,
                        w_data = NULL, b_data = NULL,
                        pads = c(0L, 0L, 0L, 0L)) {
  OH <- H - KH + 1L + pads[1] + pads[3]
  OW <- W - KW + 1L + pads[2] + pads[4]
  if (is.null(w_data)) w_data <- rep(1.0 / (C * KH * KW), OC * C * KH * KW)
  inp <- .onnx_value_info("X", 1L, c(N, C, H, W))
  outp <- .onnx_value_info("Y", 1L, c(N, OC, OH, OW))
  w_raw <- unlist(lapply(w_data, .float_bytes))
  w_t  <- .onnx_tensor("W", c(OC, C, KH, KW), 1L, w_raw)
  w_vi <- .onnx_value_info("W", 1L, c(OC, C, KH, KW))
  inputs <- list("X", "W")
  inits  <- list(w_t)
  graph_ins <- list(inp, w_vi)
  if (!is.null(b_data)) {
    b_raw <- unlist(lapply(b_data, .float_bytes))
    b_t  <- .onnx_tensor("B", c(OC), 1L, b_raw)
    b_vi <- .onnx_value_info("B", 1L, c(OC))
    inputs <- c(inputs, list("B"))
    inits  <- c(inits, list(b_t))
    graph_ins <- c(graph_ins, list(b_vi))
  }
  attrs <- list(.onnx_attr_ints("pads", pads))
  node  <- .onnx_node("Conv", inputs, "Y", attrs = attrs)
  graph <- .onnx_graph("test", list(node), graph_ins, list(outp), inits)
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  list(path = path, x = seq_len(N * C * H * W) * 0.1)
}

# ── 1. ALIASING: repeated runs must return identical results ──────────────────

test_that("aliasing: MatMul repeated run [4,3]@[3,4] produces same output", {
  m_info <- make_matmul_k(4L, 3L, 4L)
  mod <- onnx_load(m_info$path, device = "cpu")
  r1 <- onnx_run(mod, list(A = m_info$a, B = m_info$b))[[1]]
  r2 <- onnx_run(mod, list(A = m_info$a, B = m_info$b))[[1]]
  r3 <- onnx_run(mod, list(A = m_info$a, B = m_info$b))[[1]]
  expect_equal(as.numeric(r1), as.numeric(r2), tolerance = 1e-6)
  expect_equal(as.numeric(r1), as.numeric(r3), tolerance = 1e-6)
})

test_that("aliasing: Conv 3x3 repeated run [1,1,5,5] produces same output", {
  cv <- make_conv2d(1L, 1L, 5L, 5L, 1L, 3L, 3L)
  mod <- onnx_load(cv$path, device = "cpu")
  r1 <- onnx_run(mod, list(X = cv$x))[[1]]
  r2 <- onnx_run(mod, list(X = cv$x))[[1]]
  expect_equal(as.numeric(r1), as.numeric(r2), tolerance = 1e-6)
})

test_that("aliasing: Gemm repeated run with bias stays stable", {
  path <- .onnx_make_gemm(M = 3L, K = 4L, N = 5L,
                           weight_data = rep(1.0, 20),
                           bias_data = rep(0.5, 5L))
  a <- seq_len(12) * 0.1
  mod <- onnx_load(path, device = "cpu")
  r1 <- onnx_run(mod, list(A = a))[[1]]
  r2 <- onnx_run(mod, list(A = a))[[1]]
  expect_equal(as.numeric(r1), as.numeric(r2), tolerance = 1e-6)
})

test_that("aliasing: chain Relu→Sigmoid repeated run [1,8]", {
  path <- .onnx_make_chain("Relu", "Sigmoid", dims = c(1L, 8L))
  mod  <- onnx_load(path, device = "cpu")
  x    <- c(-3, -1, 0, 0.5, 1, 2, 3, 4)
  r1 <- onnx_run(mod, list(X = x))[[1]]
  r2 <- onnx_run(mod, list(X = x))[[1]]
  expect_equal(as.numeric(r1), as.numeric(r2), tolerance = 1e-6)
})

# ── 2. TILE BOUNDARY: small / odd K and N ────────────────────────────────────

# K=1 is the worst tile boundary case — single column
test_that("tile boundary: MatMul [4,1]@[1,4] K=1", {
  m_info <- make_matmul_k(4L, 1L, 4L,
                           a_data = c(1, 2, 3, 4),
                           b_data = c(1, 2, 3, 4))
  result <- run_onnx(m_info$path, list(A = m_info$a, B = m_info$b))
  # outer product: result[i,j] = a[i]*b[j]
  expected <- outer(c(1, 2, 3, 4), c(1, 2, 3, 4))
  expect_equal(as.numeric(result), as.numeric(t(expected)), tolerance = 1e-5)
})

test_that("tile boundary: MatMul [8,16]@[16,8] K=N=16 (exact tile)", {
  m_info <- make_matmul_k(8L, 16L, 8L)
  result <- run_onnx(m_info$path, list(A = m_info$a, B = m_info$b))
  a_mat <- matrix(m_info$a, nrow = 8, ncol = 16, byrow = TRUE)
  b_mat <- matrix(m_info$b, nrow = 16, ncol = 8, byrow = TRUE)
  expected <- a_mat %*% b_mat
  expect_equal(as.numeric(result), as.numeric(t(expected)), tolerance = 1e-3)
})

test_that("tile boundary: MatMul [8,32]@[32,8] K=N=32 (double tile)", {
  m_info <- make_matmul_k(8L, 32L, 8L)
  result <- run_onnx(m_info$path, list(A = m_info$a, B = m_info$b))
  a_mat <- matrix(m_info$a, nrow = 8, ncol = 32, byrow = TRUE)
  b_mat <- matrix(m_info$b, nrow = 32, ncol = 8, byrow = TRUE)
  expected <- a_mat %*% b_mat
  expect_equal(as.numeric(result), as.numeric(t(expected)), tolerance = 1e-3)
})

test_that("tile boundary: MatMul [8,33]@[33,8] K=33 crosses tile", {
  m_info <- make_matmul_k(8L, 33L, 8L)
  result <- run_onnx(m_info$path, list(A = m_info$a, B = m_info$b))
  a_mat <- matrix(m_info$a, nrow = 8, ncol = 33, byrow = TRUE)
  b_mat <- matrix(m_info$b, nrow = 33, ncol = 8, byrow = TRUE)
  expected <- a_mat %*% b_mat
  expect_equal(as.numeric(result), as.numeric(t(expected)), tolerance = 1e-3)
})

test_that("tile boundary: Conv 3x3 odd H=7 W=3", {
  cv <- make_conv2d(1L, 1L, 7L, 3L, 2L, 3L, 3L)
  result <- run_onnx(cv$path, list(X = cv$x))
  expect_equal(length(as.numeric(result)), 2L * 5L * 1L)
})

test_that("tile boundary: Conv 1x1 C=55 (non-power-of-two channels)", {
  cv <- make_conv2d(1L, 55L, 4L, 4L, 8L, 1L, 1L)
  result <- run_onnx(cv$path, list(X = cv$x))
  expect_equal(length(as.numeric(result)), 8L * 4L * 4L)
})

test_that("tile boundary: MatMul CPU vs GPU K=1", {
  m_info <- make_matmul_k(4L, 1L, 4L,
                           a_data = c(1, 2, 3, 4),
                           b_data = c(1, 2, 3, 4))
  expect_cpu_gpu_close(m_info$path, list(A = m_info$a, B = m_info$b))
})

test_that("tile boundary: MatMul CPU vs GPU K=33", {
  m_info <- make_matmul_k(8L, 33L, 8L)
  expect_cpu_gpu_close(m_info$path, list(A = m_info$a, B = m_info$b))
})

# ── 3. 4D TRUNCATION: 5D tensors must not lose last dimension ────────────────

test_that("4D truncation: 5D MatMul [2,4,3,1,1] preserved", {
  # Flatten to [8,3] before matmul, reshape output back
  inp_a  <- .onnx_value_info("A", 1L, c(2L, 4L, 3L, 1L, 1L))
  inp_b  <- .onnx_value_info("B", 1L, c(3L, 5L))
  shape_flat_raw <- c(.int64_bytes(8L), .int64_bytes(3L))
  sf_t   <- .onnx_tensor("sf", c(2L), 7L, shape_flat_raw)
  sf_vi  <- .onnx_value_info("sf", 7L, c(2L))
  shape_out_raw <- c(.int64_bytes(2L), .int64_bytes(4L), .int64_bytes(5L),
                     .int64_bytes(1L), .int64_bytes(1L))
  so_t   <- .onnx_tensor("so", c(5L), 7L, shape_out_raw)
  so_vi  <- .onnx_value_info("so", 7L, c(5L))
  outp   <- .onnx_value_info("Y", 1L, c(2L, 4L, 5L, 1L, 1L))
  n_flat <- .onnx_node("Reshape", c("A", "sf"), "A2d")
  n_mm   <- .onnx_node("MatMul",  c("A2d", "B"), "Y2d")
  n_back <- .onnx_node("Reshape", c("Y2d", "so"), "Y")
  graph  <- .onnx_graph("test", list(n_flat, n_mm, n_back),
                         list(inp_a, inp_b, sf_vi, so_vi), list(outp),
                         list(sf_t, so_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- seq_len(24) * 0.1
  b <- seq_len(15) * 0.1
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(length(as.numeric(result)), 40L)
})

test_that("4D truncation: Relu on 5D [1,2,3,4,2] preserves all 48 elements", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 2L, 3L, 4L, 2L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L, 4L, 2L))
  node <- .onnx_node("Relu", "X", "Y")
  graph <- .onnx_graph("test", list(node), list(inp), list(outp))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(-24, 23) * 0.1
  result <- run_onnx(path, list(X = x))
  expect_equal(length(as.numeric(result)), 48L)
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

test_that("4D truncation: Add two 5D tensors [1,2,3,4,2]", {
  inp_a <- .onnx_value_info("A", 1L, c(1L, 2L, 3L, 4L, 2L))
  inp_b <- .onnx_value_info("B", 1L, c(1L, 2L, 3L, 4L, 2L))
  outp  <- .onnx_value_info("Y", 1L, c(1L, 2L, 3L, 4L, 2L))
  node  <- .onnx_node("Add", c("A", "B"), "Y")
  graph <- .onnx_graph("test", list(node), list(inp_a, inp_b), list(outp))
  path  <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  a <- seq_len(48) * 0.1
  b <- rep(1.0, 48)
  result <- run_onnx(path, list(A = a, B = b))
  expect_equal(length(as.numeric(result)), 48L)
  expect_equal(as.numeric(result), a + b, tolerance = 1e-5)
})

test_that("4D truncation: Reshape to 5D then back preserves all values [60]", {
  inp <- .onnx_value_info("X", 1L, c(60L))
  to5d_raw <- c(.int64_bytes(2L), .int64_bytes(3L), .int64_bytes(5L),
                .int64_bytes(2L), .int64_bytes(1L))
  s5d_t  <- .onnx_tensor("s5d", c(5L), 7L, to5d_raw)
  s5d_vi <- .onnx_value_info("s5d", 7L, c(5L))
  back_raw <- c(.int64_bytes(60L))
  sb_t   <- .onnx_tensor("sb", c(1L), 7L, back_raw)
  sb_vi  <- .onnx_value_info("sb", 7L, c(1L))
  outp   <- .onnx_value_info("Y", 1L, c(60L))
  n1 <- .onnx_node("Reshape", c("X", "s5d"), "X5d")
  n2 <- .onnx_node("Relu",    "X5d",         "X5r")
  n3 <- .onnx_node("Reshape", c("X5r", "sb"), "Y")
  graph <- .onnx_graph("test", list(n1, n2, n3),
                        list(inp, s5d_vi, sb_vi), list(outp),
                        list(s5d_t, sb_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- seq(-30, 29) * 0.1
  result <- run_onnx(path, list(X = x))
  expect_equal(length(as.numeric(result)), 60L)
  expect_equal(as.numeric(result), pmax(x, 0), tolerance = 1e-5)
})

# ── 4. WRONG OP DISPATCH: Conv 1×1 must use MUL_MAT fast path ────────────────
# Correctness is the proxy — if dispatch is wrong, values differ.

test_that("dispatch: Conv 1x1 [1,4,4,4] correctness (MUL_MAT path)", {
  cv <- make_conv2d(1L, 4L, 4L, 4L, 8L, 1L, 1L)
  result <- run_onnx(cv$path, list(X = cv$x))
  # 1x1 conv with all-same weights: each output pixel = mean of C inputs * C
  expect_equal(length(as.numeric(result)), 8L * 4L * 4L)
  expect_true(all(is.finite(as.numeric(result))))
})

test_that("dispatch: Conv 1x1 matches manual MatMul [1,C,H,W]", {
  C <- 3L; H <- 4L; W <- 4L; OC <- 5L
  w_data <- seq_len(OC * C) * 0.1
  cv <- make_conv2d(1L, C, H, W, OC, 1L, 1L, w_data = w_data)
  result_conv <- as.numeric(run_onnx(cv$path, list(X = cv$x)))

  # Manual reference: onnx_run returns data in ggml col-major order [W,H,OC,N].
  # x arrives as ONNX [N,C,H,W] row-major → ggml stores it as [W,H,C,N].
  # For each spatial position (w,h), output[oc] = sum_c w[oc,c] * x[c,h,w].
  # Iterate spatial positions in ggml order (w fastest, then h) to match output.
  w_mat <- matrix(w_data, nrow = OC, ncol = C, byrow = TRUE)
  # x in ggml [W,H,C,N] col-major: position (w,h,c) = x[(c-1)*H*W + (h-1)*W + w]
  x_arr <- array(cv$x, dim = c(W, H, C))  # col-major fill matches ggml layout
  expected <- apply(x_arr, c(1, 2), function(xvec) as.numeric(w_mat %*% xvec))
  # apply returns [OC, W, H]; flatten col-major → [OC,W,H] which is [W,H,OC] in ggml order
  expected <- as.numeric(aperm(array(expected, dim = c(OC, W, H)), c(2, 3, 1)))

  expect_equal(result_conv, expected, tolerance = 1e-3)
})

test_that("dispatch: Conv 3x3 vs Conv 1x1 differ on same input [1,2,5,5]", {
  cv1 <- make_conv2d(1L, 2L, 5L, 5L, 4L, 1L, 1L)
  cv3 <- make_conv2d(1L, 2L, 5L, 5L, 4L, 3L, 3L)
  r1 <- as.numeric(run_onnx(cv1$path, list(X = cv1$x)))
  r3 <- as.numeric(run_onnx(cv3$path, list(X = cv3$x)))
  expect_equal(length(r1), 4L * 5L * 5L)
  expect_equal(length(r3), 4L * 3L * 3L)
  # They must differ (different receptive fields)
  expect_true(!isTRUE(all.equal(r1[seq_along(r3)], r3, tolerance = 1e-3)))
})

test_that("dispatch: Conv 1x1 CPU vs GPU [1,8,6,6]", {
  cv <- make_conv2d(1L, 8L, 6L, 6L, 16L, 1L, 1L)
  expect_cpu_gpu_close(cv$path, list(X = cv$x), tol = 1e-2)
})

# ── 5. GRAPH CHAINS: accumulated error across 3-5 nodes ──────────────────────

test_that("graph chain: Relu→LayerNorm→MatMul [4,4] stable", {
  inp_a <- .onnx_value_info("X", 1L, c(4L, 4L))

  # LayerNorm weights: scale=1, bias=0
  sc_raw <- unlist(lapply(rep(1.0, 4), .float_bytes))
  sc_t   <- .onnx_tensor("sc", c(4L), 1L, sc_raw)
  sc_vi  <- .onnx_value_info("sc", 1L, c(4L))
  bi_raw <- unlist(lapply(rep(0.0, 4), .float_bytes))
  bi_t   <- .onnx_tensor("bi", c(4L), 1L, bi_raw)
  bi_vi  <- .onnx_value_info("bi", 1L, c(4L))

  # MatMul weight: identity
  w_data <- as.numeric(diag(4))
  w_raw  <- unlist(lapply(w_data, .float_bytes))
  w_t    <- .onnx_tensor("W", c(4L, 4L), 1L, w_raw)
  w_vi   <- .onnx_value_info("W", 1L, c(4L, 4L))

  outp <- .onnx_value_info("Y", 1L, c(4L, 4L))

  n1 <- .onnx_node("Relu",               "X",             "r1")
  n2 <- .onnx_node("LayerNormalization", c("r1", "sc", "bi"), "r2",
                   attrs = list(.onnx_attr_float("epsilon", 1e-5)))
  n3 <- .onnx_node("MatMul",             c("r2", "W"),    "Y")

  graph <- .onnx_graph("test", list(n1, n2, n3),
                        list(inp_a, sc_vi, bi_vi, w_vi), list(outp),
                        list(sc_t, bi_t, w_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-2, -1, 1, 2, 3, 4, -3, 0, 1, -1, 2, 3, 0, 1, 2, 3)
  result <- run_onnx(path, list(X = x))
  expect_equal(length(as.numeric(result)), 16L)
  expect_true(all(is.finite(as.numeric(result))))
})

test_that("graph chain: Split→Relu→Concat round-trip [1,8]", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 8L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 8L))

  # Split [1,8] → s0[1,4] + s1[1,4]
  s0_vi <- .onnx_value_info("s0", 1L, c(1L, 4L))
  s1_vi <- .onnx_value_info("s1", 1L, c(1L, 4L))
  r0_vi <- .onnx_value_info("r0", 1L, c(1L, 4L))
  r1_vi <- .onnx_value_info("r1", 1L, c(1L, 4L))

  split_raw <- c(.int64_bytes(4L), .int64_bytes(4L))
  split_t   <- .onnx_tensor("split_sz", c(2L), 7L, split_raw)
  split_vi  <- .onnx_value_info("split_sz", 7L, c(2L))

  n_split  <- .onnx_node("Split",  c("X", "split_sz"), c("s0", "s1"),
                          attrs = list(.onnx_attr_int("axis", 1L)))
  n_relu0  <- .onnx_node("Relu",   "s0", "r0")
  n_relu1  <- .onnx_node("Relu",   "s1", "r1")
  n_concat <- .onnx_node("Concat", c("r0", "r1"), "Y",
                          attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
                        list(n_split, n_relu0, n_relu1, n_concat),
                        list(inp, split_vi), list(outp),
                        list(split_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-4, -3, -2, -1, 1, 2, 3, 4)
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(length(result), 8L)
  expect_equal(result, pmax(x, 0), tolerance = 1e-5)
})

test_that("graph chain: Split→Relu→Concat last output not aliased [1,12]", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 12L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 12L))

  # Split into 3 chunks of 4
  split_raw <- c(.int64_bytes(4L), .int64_bytes(4L), .int64_bytes(4L))
  split_t   <- .onnx_tensor("ss", c(3L), 7L, split_raw)
  split_vi  <- .onnx_value_info("ss", 7L, c(3L))

  n_split  <- .onnx_node("Split",  c("X", "ss"), c("s0", "s1", "s2"),
                          attrs = list(.onnx_attr_int("axis", 1L)))
  n_r0 <- .onnx_node("Relu", "s0", "r0")
  n_r1 <- .onnx_node("Relu", "s1", "r1")
  n_r2 <- .onnx_node("Relu", "s2", "r2")
  n_cc <- .onnx_node("Concat", c("r0", "r1", "r2"), "Y",
                      attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
                        list(n_split, n_r0, n_r1, n_r2, n_cc),
                        list(inp, split_vi), list(outp),
                        list(split_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-6:-1, 1:6) * 1.0
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(length(result), 12L)
  # last chunk (indices 9-12) must not be aliased to first chunk
  expect_equal(result, pmax(x, 0), tolerance = 1e-5)
})

test_that("graph chain: Split 4 outputs last output correct [1,16]", {
  inp  <- .onnx_value_info("X", 1L, c(1L, 16L))
  outp <- .onnx_value_info("Y", 1L, c(1L, 16L))

  split_raw <- c(.int64_bytes(4L), .int64_bytes(4L),
                 .int64_bytes(4L), .int64_bytes(4L))
  split_t   <- .onnx_tensor("ss4", c(4L), 7L, split_raw)
  split_vi  <- .onnx_value_info("ss4", 7L, c(4L))

  n_split <- .onnx_node("Split", c("X", "ss4"), c("s0","s1","s2","s3"),
                         attrs = list(.onnx_attr_int("axis", 1L)))
  n_r <- lapply(0:3, function(i)
    .onnx_node("Relu", paste0("s", i), paste0("r", i)))
  n_cc <- .onnx_node("Concat", c("r0","r1","r2","r3"), "Y",
                      attrs = list(.onnx_attr_int("axis", 1L)))

  graph <- .onnx_graph("test",
                        c(list(n_split), n_r, list(n_cc)),
                        list(inp, split_vi), list(outp),
                        list(split_t))
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)

  x <- c(-8:-1, 1:8) * 1.0
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(length(result), 16L)
  expect_equal(result, pmax(x, 0), tolerance = 1e-5)
})

# ── 6. BATCH DIMENSION: batch=1 vs batch>1 ───────────────────────────────────

test_that("batch: MatMul batch=1 [1,4,3]@[3,5] correct", {
  # Treated as 2D [4,3]@[3,5] by ggml
  m_info <- make_matmul_k(4L, 3L, 5L)
  result <- as.numeric(run_onnx(m_info$path, list(A = m_info$a, B = m_info$b)))
  a_mat <- matrix(m_info$a, nrow = 4, ncol = 3, byrow = TRUE)
  b_mat <- matrix(m_info$b, nrow = 3, ncol = 5, byrow = TRUE)
  expected <- as.numeric(t(a_mat %*% b_mat))
  expect_equal(result, expected, tolerance = 1e-3)
})

test_that("batch: Relu batch=4 [4,8] all elements processed", {
  path <- .onnx_make_unary("Relu", c(4L, 8L))
  x <- c(seq(-16, -1), seq(1, 16)) * 0.1
  result <- as.numeric(run_onnx(path, list(X = x)))
  expect_equal(length(result), 32L)
  expect_equal(result, pmax(x, 0), tolerance = 1e-5)
})

test_that("batch: Conv batch=2 [2,1,5,5] processes both images", {
  cv <- make_conv2d(2L, 1L, 5L, 5L, 2L, 3L, 3L)
  result <- as.numeric(run_onnx(cv$path, list(X = cv$x)))
  expect_equal(length(result), 2L * 2L * 3L * 3L)
  expect_true(all(is.finite(result)))
})
