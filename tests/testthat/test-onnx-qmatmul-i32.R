# QLinearMatMul integer path (GGML_OP_QMATMUL_I32), CPU and Vulkan, checked
# bit for bit against an R reference of the same arithmetic: int16-saturated
# pairs along K (VPMADDUBSW), zero points folded in at the end, per-column
# weight scale, round-half-even, uint8 clamp.
#
# Operands span the full uint8 / int8 range so pairs genuinely saturate, and
# the test asserts that they do: without saturation the exact f32 fallback
# would pass as well and the test would prove nothing.

qmm_run <- function(path, inputs, device) {
  m <- onnx_load(path, device = device)
  onnx_run(m, inputs)[[1]]
}

# A is ONNX [M, K] (uint8 values), B is [K, N] (int8 values).
qmm_ref_acc <- function(A, B, a_zp, b_zp) {
  M <- nrow(A); K <- ncol(A); N <- ncol(B)
  acc <- matrix(0, M, N)
  n_sat <- 0L
  for (m in seq_len(M)) for (n in seq_len(N)) {
    s <- 0
    for (k in seq(1, K, by = 2)) {
      a1 <- A[m, k]; b1 <- B[k, n]
      a2 <- if (k + 1 <= K) A[m, k + 1] else 0
      b2 <- if (k + 1 <= K) B[k + 1, n] else 0
      pair <- a1 * b1 + a2 * b2
      if (pair > 32767 || pair < -32768) n_sat <- n_sat + 1L
      s <- s + min(max(pair, -32768), 32767)
    }
    bz <- b_zp[if (length(b_zp) > 1) n else 1]
    acc[m, n] <- s - bz * sum(A[m, ]) - a_zp * sum(B[, n]) + a_zp * bz * K
  }
  list(acc = acc, n_sat = n_sat)
}

qmm_model <- function(M, K, N, B, b_scale, b_zp, a_zp, y_scale, y_zp) {
  f32 <- function(nm, v) .onnx_tensor(nm, length(v), 1L,
                                      unlist(lapply(v, .float_bytes)))
  u8  <- function(nm, v) .onnx_tensor(nm, 1L, 2L, as.raw(v))
  i8  <- function(nm, dims, v) .onnx_tensor(nm, dims, 3L,
                                            as.raw(ifelse(v < 0, v + 256, v)))
  inits <- list(
    f32("a_sc", 1), u8("a_zp", a_zp),
    # B row-major [K, N]: t(B) flattens it row by row.
    i8("B", c(K, N), as.vector(t(B))),
    f32("b_sc", b_scale),
    f32("y_sc", y_scale), u8("y_zp", y_zp))
  ins <- c("A", "a_sc", "a_zp", "B", "b_sc")
  if (!is.null(b_zp)) {
    inits <- c(inits, list(i8("b_zp", length(b_zp), b_zp)))
    ins <- c(ins, "b_zp")
  } else {
    ins <- c(ins, "")   # optional input left empty
  }
  ins <- c(ins, "y_sc", "y_zp")
  node  <- .onnx_node("QLinearMatMul", ins, "Y")
  graph <- .onnx_graph("qmatmul_i32", list(node),
    list(.onnx_value_info("A", 1L, c(M, K))),
    list(.onnx_value_info("Y", 1L, c(M, N))),
    inits)
  path <- tempfile(fileext = ".onnx")
  writeBin(.onnx_model(graph), path)
  path
}

check_qmm_case <- function(M, K, N, seed, per_col = TRUE, b_zp_mode = "zero",
                           a_zp = 7, device = "cpu") {
  set.seed(seed)
  A <- matrix(sample(0:255, M * K, replace = TRUE), M, K)
  B <- matrix(sample(-128:127, K * N, replace = TRUE), K, N)
  # Guarantee saturating pairs: row 1 of A against column 1 of B at the
  # extremes gives 255*127*2 = 64770 per pair.
  A[1, ] <- 255
  B[, 1] <- 127
  b_zp <- switch(b_zp_mode,
                 none   = NULL,
                 zero   = rep(0, N),
                 nonzero = sample(-5:5, N, replace = TRUE))
  ref <- qmm_ref_acc(A, B, a_zp, if (is.null(b_zp)) 0 else b_zp)
  expect_gt(ref$n_sat, 0L)

  # Powers of two only, so a_sc * b_sc[n] / y_sc and acc * mult are exact in
  # float and the reference needs no float emulation; y_sc keeps the outputs
  # inside [0, 255] so every element is informative.
  b_scale <- if (per_col) 2^sample(-1:1, N, replace = TRUE) else 1
  scaled  <- sweep(ref$acc, 2, rep_len(b_scale, N), `*`)
  y_scale <- 2^ceiling(log2(max(abs(scaled)) / 100))
  y_zp    <- 128

  path <- qmm_model(M, K, N, B, b_scale, b_zp, a_zp, y_scale, y_zp)
  # A row-major [M, K]: t(A) flattens it row by row.
  got  <- as.numeric(qmm_run(path, list(A = as.numeric(t(A))), device))
  # Output row-major [M, N].
  want <- as.numeric(t(pmin(pmax(round(scaled / y_scale) + y_zp, 0), 255)))
  expect_equal(length(got), length(want))
  expect_identical(got, want)
}

for (dev in c("cpu", "vulkan")) {
  local({
    device <- dev
    skip_dev <- function() {
      if (device == "vulkan")
        skip_if_not(ggml_vulkan_available(), "Vulkan not available")
    }
    test_that(sprintf("qmatmul_i32 %s: odd K, per-column scale, zero b_zp", device), {
      skip_dev()
      check_qmm_case(M = 5, K = 33, N = 7, seed = 1, device = device)
    })
    test_that(sprintf("qmatmul_i32 %s: non-zero b_zp and a_zp (b_zp_any)", device), {
      skip_dev()
      check_qmm_case(M = 6, K = 40, N = 9, seed = 2,
                     b_zp_mode = "nonzero", device = device)
    })
    test_that(sprintf("qmatmul_i32 %s: no b_zp input (stand-in binding)", device), {
      skip_dev()
      check_qmm_case(M = 4, K = 17, N = 5, seed = 3,
                     b_zp_mode = "none", device = device)
    })
    test_that(sprintf("qmatmul_i32 %s: per-tensor scale, tile-crossing shape", device), {
      skip_dev()
      # M and N past one 16x16 block, K past one TILE_K (128) and odd.
      check_qmm_case(M = 19, K = 259, N = 21, seed = 4, per_col = FALSE,
                     device = device)
    })
  })
}
