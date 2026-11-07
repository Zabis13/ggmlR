#!/usr/bin/env Rscript
# Timings of the CURRENT ggmlR build -- no comparisons against other versions.
#
# Sections:
#   ops       one-op graphs on each backend in BACKENDS: conv_2d_direct (incl.
#             the 64x32 tile), im2col, leaky_relu / gelu / silu, soft_max,
#             out_prod, get_rows_back, cross_entropy_loss_back, top_k,
#             mul_mat (F32 / F16 / Q8_0 / Q4_K, mat-vec and mat-mat)
#   maskrcnn  onnx_run of MaskRCNN-12-int8: load, warm-up, N timed runs,
#             process peak RSS and device memory in use
#
# Usage:
#   Rscript bench_vk_current.R ops
#   Rscript bench_vk_current.R maskrcnn vulkan
#   Rscript bench_vk_current.R maskrcnn cpu
# (run maskrcnn once per device, in its own process: peak RSS is per process)
#
# Each op graph is built and its data uploaded ONCE; only
# ggml_backend_graph_compute() + synchronize is timed. One-op graphs on the
# GPU include one submit + fence wait per call, so for small shapes the time is
# that overhead, not the shader -- the "small" rows show where that floor is.

suppressMessages(library(ggmlR))

# ---- parameters (edit here) -------------------------------------------------
N_WARM    <- 3L          # untimed computes before timing
N_REP     <- 20L         # timed computes per case (median reported)
BACKENDS  <- c("vulkan", "cpu")
CPU_THREADS <- 8L
OPS       <- c("conv2d", "im2col", "unary", "softmax", "out_prod",
               "get_rows_back", "ce_loss_back", "top_k", "mul_mat")

MRCNN_DIR   <- Sys.getenv("ONNX_DIR", "/mnt/Data2/DS_projects/ONNX models-main")
MRCNN_FILE  <- "MaskRCNN-12-int8.onnx"
MRCNN_SHAPE <- c(3L, 224L, 224L)
MRCNN_RUNS  <- 10L
# -----------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
SECTION <- if (length(args) >= 1) args[1] else "ops"
MDEV    <- if (length(args) >= 2) args[2] else "vulkan"

if ("vulkan" %in% BACKENDS && !ggml_vulkan_available()) {
  cat("Vulkan not available -- ops run on CPU only\n")
  BACKENDS <- setdiff(BACKENDS, "vulkan")
}

# Sys.time(): microsecond resolution on Linux (proc.time() elapsed is 1 ms)
t_ms <- function(f) {
  t0 <- Sys.time()
  f()
  as.numeric(Sys.time() - t0, units = "secs") * 1000
}

# inputs: list of list(type, shape, data). build(ctx, tensors...) -> out tensor.
# Returns median / min ms of the compute alone.
bench_graph <- function(backend_name, inputs, build) {
  ctx <- ggml_init(64 * 1024 * 1024, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)
  tensors <- lapply(inputs, function(x) {
    ne <- c(x$shape, rep(1, 4 - length(x$shape)))
    ggml_new_tensor_4d(ctx, x$type, ne[1], ne[2], ne[3], ne[4])
  })
  out <- do.call(build, c(list(ctx), tensors))
  gf  <- ggml_build_forward_expand(ctx, out)
  backend <- if (backend_name == "vulkan") ggml_vulkan_init(0) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE, after = FALSE)
  if (backend_name == "cpu") ggml_backend_cpu_set_n_threads(backend, CPU_THREADS)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE, after = FALSE)
  for (i in seq_along(inputs)) ggml_backend_tensor_set_data(tensors[[i]], inputs[[i]]$data)

  run <- function() { ggml_backend_graph_compute(backend, gf); ggml_backend_synchronize(backend) }
  for (i in seq_len(N_WARM)) run()
  ts <- vapply(seq_len(N_REP), function(i) t_ms(run), numeric(1))
  c(med = median(ts), min = min(ts))
}

f32 <- function(n, ...) list(type = GGML_TYPE_F32, shape = c(...), data = rnorm(n))
f16 <- function(n, ...) list(type = GGML_TYPE_F16, shape = c(...), data = rnorm(n))
i32 <- function(v, ...) list(type = GGML_TYPE_I32, shape = c(...), data = as.integer(v))
raw_q <- function(type, qfun, k, m) list(type = type, shape = c(k, m), data = qfun(rnorm(k * m), m, k))

results <- list()
report <- function(op, case, backend, r, work = NA_real_, unit = "") {
  rate <- if (is.na(work)) "" else sprintf("%9.1f %s", work / (r[["med"]] / 1000) / 1e9, unit)
  cat(sprintf("  %-14s %-34s %-6s %9.3f ms (min %8.3f) %s\n",
              op, case, backend, r[["med"]], r[["min"]], rate))
  flush.console()
  results[[length(results) + 1]] <<- data.frame(op = op, case = case, backend = backend,
                                                med_ms = r[["med"]], min_ms = r[["min"]],
                                                stringsAsFactors = FALSE)
}

each_backend <- function(op, case, inputs, build, work = NA_real_, unit = "") {
  for (b in BACKENDS) {
    r <- tryCatch(bench_graph(b, inputs, build),
                  error = function(e) { cat(sprintf("  %-14s %-34s %-6s ERROR: %s\n",
                                                    op, case, b, conditionMessage(e))); NULL })
    if (!is.null(r)) report(op, case, b, r, work, unit)
  }
}

# ---- ops ---------------------------------------------------------------------

run_ops <- function() {
  set.seed(1)
  cat(sprintf("=== ops: warm %d, rep %d, backends %s ===\n",
              N_WARM, N_REP, paste(BACKENDS, collapse = ",")))
  if ("vulkan" %in% BACKENDS) {
    m <- ggml_vulkan_device_memory(0)
    cat(sprintf("Vulkan device 0, %.1f GB total\n", m$total / 1e9))
  }

  if ("conv2d" %in% OPS) {
    cat("\n-- conv_2d_direct (F32 kernel, s1 p1) -- GFLOP/s\n")
    # KW KH W H Cin Cout N ; small ones pick CONV_SHAPE_64x32
    for (cf in list(c(3, 3, 6, 6, 4, 8, 1), c(3, 3, 14, 14, 32, 64, 1),
                    c(3, 3, 28, 28, 64, 64, 1), c(3, 3, 56, 56, 64, 128, 1),
                    c(3, 3, 112, 112, 64, 64, 4), c(1, 1, 56, 56, 256, 64, 8))) {
      KW <- cf[1]; KH <- cf[2]; W <- cf[3]; H <- cf[4]; Ci <- cf[5]; Co <- cf[6]; N <- cf[7]
      flop <- 2 * KW * KH * Ci * Co * W * H * N
      each_backend("conv2d", sprintf("k%dx%d %dx%d c%d->%d n%d", KW, KH, W, H, Ci, Co, N),
                   list(f32(KW * KH * Ci * Co, KW, KH, Ci, Co), f32(W * H * Ci * N, W, H, Ci, N)),
                   function(ctx, k, x) ggml_conv_2d_direct(ctx, k, x, 1L, 1L, 1L, 1L, 1L, 1L),
                   flop, "GFLOP/s")
    }
  }

  if ("im2col" %in% OPS) {
    cat("\n-- im2col 2D (s1 p1), F32 and F16 out -- GB/s written\n")
    # KW KH W H C N
    for (cf in list(c(3, 3, 8, 8, 1024, 2), c(3, 3, 56, 56, 64, 4),
                    c(3, 3, 224, 224, 3, 8), c(7, 7, 112, 112, 64, 1))) {
      KW <- cf[1]; KH <- cf[2]; W <- cf[3]; H <- cf[4]; C <- cf[5]; N <- cf[6]
      n_out <- KW * KH * C * W * H * N
      for (dt in c("F32", "F16")) {
        type <- if (dt == "F32") GGML_TYPE_F32 else GGML_TYPE_F16
        each_backend("im2col", sprintf("k%dx%d %dx%d c%d n%d %s", KW, KH, W, H, C, N, dt),
                     list(f32(KW * KH * C, KW, KH, C, 1), f32(W * H * C * N, W, H, C, N)),
                     function(ctx, a, b) ggml_im2col(ctx, a, b, 1L, 1L, 1L, 1L, 1L, 1L,
                                                     is_2D = TRUE, dst_type = type),
                     n_out * if (dt == "F32") 4 else 2, "GB/s")
      }
    }
  }

  if ("unary" %in% OPS) {
    cat("\n-- unary F32 -- GB/s (read + write)\n")
    ops <- list(leaky_relu = function(ctx, a) ggml_leaky_relu(ctx, a, 0.1),
                gelu = ggml_gelu, silu = ggml_silu)
    for (n in c(4096L, 1048576L, 16777216L, 512L * 65536L + 7L)) {
      inp <- list(f32(n, n))
      for (nm in names(ops)) each_backend(nm, sprintf("n=%d", n), inp, ops[[nm]], 8 * n, "GB/s")
    }
  }

  if ("softmax" %in% OPS) {
    cat("\n-- soft_max F32 -- GB/s\n")
    for (cf in list(c(128, 4096), c(4096, 512), c(32000, 32))) {
      n <- cf[1] * cf[2]
      each_backend("soft_max", sprintf("%dx%d", cf[1], cf[2]), list(f32(n, cf[1], cf[2])),
                   ggml_soft_max, 8 * n, "GB/s")
    }
  }

  if ("out_prod" %in% OPS) {
    cat("\n-- out_prod F32: a[m,k] x b[n,k] -> [m,n] -- GFLOP/s\n")
    for (cf in list(c(64, 64, 64), c(512, 512, 128), c(1024, 1024, 256), c(4096, 256, 512))) {
      m <- cf[1]; n <- cf[2]; k <- cf[3]
      each_backend("out_prod", sprintf("m%d n%d k%d", m, n, k),
                   list(f32(m * k, m, k), f32(n * k, n, k)), ggml_out_prod,
                   2 * m * n * k, "GFLOP/s")
    }
  }

  if ("get_rows_back" %in% OPS) {
    cat("\n-- get_rows_back F32: grad[d, n_idx] -> [d, vocab] (repeated ids)\n")
    for (cf in list(c(64, 256, 1000), c(256, 4096, 30000), c(768, 8192, 50000))) {
      d <- cf[1]; ni <- cf[2]; V <- cf[3]
      each_backend("get_rows_back", sprintf("d%d idx%d vocab%d", d, ni, V),
                   list(f32(d * ni, d, ni), i32(sample.int(V, ni, replace = TRUE) - 1L, ni),
                        f32(d * V, d, V)),
                   ggml_get_rows_back)
    }
  }

  if ("ce_loss_back" %in% OPS) {
    cat("\n-- cross_entropy_loss_back F32: logits[C, N] -- GB/s (3 reads + write)\n")
    for (cf in list(c(10, 1024), c(1000, 256), c(32000, 64))) {
      C <- cf[1]; N <- cf[2]; n <- C * N
      lab <- matrix(0, C, N); lab[cbind(sample.int(C, N, replace = TRUE), seq_len(N))] <- 1
      each_backend("ce_loss_back", sprintf("C%d N%d", C, N),
                   list(list(type = GGML_TYPE_F32, shape = 1, data = 1),
                        f32(n, C, N), list(type = GGML_TYPE_F32, shape = c(C, N), data = as.numeric(lab))),
                   ggml_cross_entropy_loss_back, 12 * n, "GB/s")
    }
  }

  if ("top_k" %in% OPS) {
    cat("\n-- top_k F32: rows of n, k\n")
    for (cf in list(c(1024, 64, 8), c(32000, 8, 40), c(151936, 1, 64), c(4096, 512, 256))) {
      n <- cf[1]; r <- cf[2]; k <- cf[3]
      each_backend("top_k", sprintf("n%d rows%d k%d", n, r, k), list(f32(n * r, n, r)),
                   function(ctx, a) ggml_top_k(ctx, a, as.integer(k)))
    }
  }

  if ("mul_mat" %in% OPS) {
    cat("\n-- mul_mat: W[K,M] (type) x X[K,N] F32 -- GFLOP/s\n")
    K <- 4096L; M <- 4096L
    wts <- list(F32  = function() f32(K * M, K, M),
                F16  = function() f16(K * M, K, M),
                Q8_0 = function() raw_q(GGML_TYPE_Q8_0, quantize_q8_0, K, M),
                Q4_K = function() raw_q(GGML_TYPE_Q4_K, quantize_q4_K, K, M))
    for (wt in names(wts)) {
      w <- wts[[wt]]()
      for (N in c(1L, 8L, 512L)) {
        each_backend("mul_mat", sprintf("%s K%d M%d N%d", wt, K, M, N),
                     list(w, f32(K * N, K, N)), ggml_mul_mat, 2 * K * M * N, "GFLOP/s")
      }
    }
  }

  df <- do.call(rbind, results)
  out <- file.path("/tmp", sprintf("bench_vk_current_ops_%s.csv", format(Sys.time(), "%Y%m%d_%H%M%S")))
  write.csv(df, out, row.names = FALSE)
  cat(sprintf("\nsaved %s\n", out))
}

# ---- maskrcnn ----------------------------------------------------------------

vm_kb <- function(key) {
  l <- grep(paste0("^", key, ":"), readLines("/proc/self/status"), value = TRUE)
  if (!length(l)) return(NA_real_)
  as.numeric(gsub("[^0-9]", "", l))
}

run_maskrcnn <- function(dev) {
  path <- file.path(MRCNN_DIR, MRCNN_FILE)
  if (!file.exists(path)) stop("model not found: ", path)
  cat(sprintf("=== MaskRCNN: device=%s, %d timed runs ===\n", dev, MRCNN_RUNS))
  vk <- dev == "vulkan" && ggml_vulkan_available()
  mem0 <- if (vk) ggml_vulkan_device_memory(0)$free else NA

  t_load <- t_ms(function()
    model <<- onnx_load(path, device = dev, input_shapes = list(image = MRCNN_SHAPE)))
  set.seed(42)
  inp <- list(image = runif(prod(MRCNN_SHAPE)))

  out <- NULL
  t_first <- t_ms(function() out <<- onnx_run(model, inp))
  n_det <- length(as.numeric(out[[1]])) %/% 4L
  ts <- vapply(seq_len(MRCNN_RUNS), function(i) t_ms(function() out <<- onnx_run(model, inp)),
               numeric(1))
  n_det_last <- length(as.numeric(out[[1]])) %/% 4L
  mem1 <- if (vk) ggml_vulkan_device_memory(0)$free else NA

  cat(sprintf("load        %8.1f ms\n", t_load))
  cat(sprintf("first run   %8.1f ms  (detections %d)\n", t_first, n_det))
  cat(sprintf("runs        median %8.1f ms  min %8.1f  max %8.1f  (detections last %d)\n",
              median(ts), min(ts), max(ts), n_det_last))
  cat(sprintf("peak RSS    %8.1f MB  (VmHWM)\n", vm_kb("VmHWM") / 1024))
  if (vk) cat(sprintf("device mem  %8.1f MB in use after runs (free before load - free now)\n",
                      (mem0 - mem1) / 2^20))
}

# ---- main ----------------------------------------------------------------------

switch(SECTION,
       ops      = run_ops(),
       maskrcnn = run_maskrcnn(MDEV),
       stop("unknown section: ", SECTION, " (ops | maskrcnn)"))
