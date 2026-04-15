#!/usr/bin/env Rscript
# Op-level micro-benchmark: every major ggml op on CPU and GPU (Vulkan)
# Run: Rscript inst/examples/benchmark_ops.R
#
# Each op is benchmarked with N_WARMUP warmup runs + N_RUNS timed runs.
# Output: per-op table with mean/min/max ms, GFLOPS where applicable,
# and GPU/CPU speedup.

Sys.setenv(GGML_VK_PERF_LOGGER = "0")   # silence per-kernel timing spam

library(ggmlR)

N_WARMUP    <- 0L
N_RUNS      <- 1L
TARGET_MS   <- 10.0   # target min total time per timed run (ms)
              # bench_op auto-repeats graph until elapsed >= TARGET_MS,
              # so proc.time() resolution (~1ms) never dominates.

# ---- helpers ----------------------------------------------------------------

make_backend <- function(device) {
  if (device == "cpu") {
    nc <- parallel::detectCores(logical = FALSE)
    if (is.na(nc)) nc <- 1L
    b <- ggml_backend_cpu_init()
    ggml_backend_cpu_set_n_threads(b, max(nc - 1L, 1L))
    b
  } else {
    ggml_vulkan_init(0L)
  }
}

# Allocate context + tensors, build graph, run, return timing list.
# `build_fn(ctx)` must return the output tensor.
# Auto-batching: first does a calibration run to find n_batch so that
# one timed sample takes >= TARGET_MS, then divides back to per-op time.
bench_op <- function(device, build_fn,
                     n_warmup = N_WARMUP, n_runs = N_RUNS) {
  tryCatch({
    ctx <- ggml_init(mem_size = 256L * 1024L * 1024L, no_alloc = TRUE)
    out <- build_fn(ctx)
    if (is.null(out)) { ggml_free(ctx); return(NULL) }

    backend <- make_backend(device)
    buf     <- ggml_backend_alloc_ctx_tensors(ctx, backend)
    graph   <- ggml_build_forward_expand(ctx, out)

    # warmup
    for (i in seq_len(n_warmup)) ggml_backend_graph_compute(backend, graph)

    # calibration: find n_batch so total >= TARGET_MS
    n_batch <- 1L
    t0 <- proc.time()
    ggml_backend_graph_compute(backend, graph)
    single_ms <- (proc.time() - t0)[3] * 1e3
    if (single_ms < TARGET_MS) {
      n_batch <- max(1L, as.integer(ceiling(TARGET_MS / max(single_ms, 0.001))))
    }

    # timed runs
    times <- numeric(n_runs)
    for (i in seq_len(n_runs)) {
      t0 <- proc.time()
      for (j in seq_len(n_batch)) ggml_backend_graph_compute(backend, graph)
      times[i] <- (proc.time() - t0)[3] * 1e3 / n_batch  # per-op ms
    }

    ggml_backend_buffer_free(buf)
    ggml_backend_free(backend)
    ggml_free(ctx)

    list(mean_ms  = mean(times),
         min_ms   = min(times),
         max_ms   = max(times),
         sd_ms    = sd(times),
         n_batch  = n_batch)
  }, error = function(e) {
    message("  ERROR: ", e$message)
    NULL
  })
}

# ---- op registry ------------------------------------------------------------
# Each entry: name, flops_fn(sizes) or NULL, build_fn(ctx, sizes)
# sizes: named list passed to build_fn for parameterisation

make_ops <- function() {
  list(

    # ------------------------------------------------------------------
    # Elementwise unary  (N = 4M floats)
    # ------------------------------------------------------------------
    list(name = "RELU",    sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_relu(ctx, a) }),

    list(name = "GELU",    sz = list(N=4e6),
         flops = function(s) s$N * 8,   # approx tanh path
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_gelu(ctx, a) }),

    list(name = "SILU",    sz = list(N=4e6),
         flops = function(s) s$N * 4,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_silu(ctx, a) }),

    list(name = "TANH",    sz = list(N=4e6),
         flops = function(s) s$N * 8,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_tanh(ctx, a) }),

    list(name = "EXP",     sz = list(N=4e6),
         flops = function(s) s$N * 4,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_exp(ctx, a) }),

    list(name = "SQRT",    sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_sqrt(ctx, a) }),

    list(name = "SQR",     sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_sqr(ctx, a) }),

    list(name = "ABS",     sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_abs(ctx, a) }),

    list(name = "NEG",     sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_neg(ctx, a) }),

    list(name = "LOG",     sz = list(N=4e6),
         flops = function(s) s$N * 4,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_log(ctx, a) }),

    list(name = "SIN",     sz = list(N=4e6),
         flops = function(s) s$N * 8,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_sin(ctx, a) }),

    list(name = "COS",     sz = list(N=4e6),
         flops = function(s) s$N * 8,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_cos(ctx, a) }),

    # ------------------------------------------------------------------
    # Elementwise binary  (N = 4M)
    # ------------------------------------------------------------------
    list(name = "ADD",     sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_add(ctx, a, b) }),

    list(name = "MUL",     sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_mul(ctx, a, b) }),

    list(name = "DIV",     sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_div(ctx, a, b) }),

    # ------------------------------------------------------------------
    # SCALE (broadcast scalar)
    # ------------------------------------------------------------------
    list(name = "SCALE",   sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_scale(ctx, a, 0.5) }),

    # ------------------------------------------------------------------
    # MUL_MAT — several shapes
    # ------------------------------------------------------------------
    list(name = "MUL_MAT 1024x1024x1024", sz = list(M=1024L, N=1024L, K=1024L),
         flops = function(s) 2 * s$M * s$N * s$K,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$M))
           b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$N))
           ggml_mul_mat(ctx, a, b) }),

    list(name = "MUL_MAT 4096x4096x256", sz = list(M=4096L, N=4096L, K=256L),
         flops = function(s) 2 * s$M * s$N * s$K,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$M))
           b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$N))
           ggml_mul_mat(ctx, a, b) }),

    list(name = "MUL_MAT 512x512x512",   sz = list(M=512L, N=512L, K=512L),
         flops = function(s) 2 * s$M * s$N * s$K,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$M))
           b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$N))
           ggml_mul_mat(ctx, a, b) }),

    list(name = "MUL_MAT 128x4096x4096", sz = list(M=128L, N=4096L, K=4096L),
         flops = function(s) 2 * s$M * s$N * s$K,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$M))
           b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$N))
           ggml_mul_mat(ctx, a, b) }),

    list(name = "MUL_MAT 1x4096x4096",   sz = list(M=1L, N=4096L, K=4096L),
         flops = function(s) 2 * s$M * s$N * s$K,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$M))
           b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$K), as.integer(s$N))
           ggml_mul_mat(ctx, a, b) }),

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    list(name = "NORM",      sz = list(rows=1024L, cols=1024L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_norm(ctx, a) }),

    list(name = "RMS_NORM",  sz = list(rows=1024L, cols=1024L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_rms_norm(ctx, a) }),

    # ------------------------------------------------------------------
    # SOFT_MAX — sweep around wg512 threshold (>= 512 → wg512)
    # ------------------------------------------------------------------
    list(name = "SOFT_MAX 128x1024",  sz = list(rows=1024L, cols=128L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_soft_max(ctx, a) }),

    list(name = "SOFT_MAX 256x1024",  sz = list(rows=1024L, cols=256L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_soft_max(ctx, a) }),

    list(name = "SOFT_MAX 512x1024",  sz = list(rows=1024L, cols=512L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_soft_max(ctx, a) }),

    list(name = "SOFT_MAX 1024x1024", sz = list(rows=1024L, cols=1024L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_soft_max(ctx, a) }),

    list(name = "SOFT_MAX 2048x1024", sz = list(rows=1024L, cols=2048L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_soft_max(ctx, a) }),

    list(name = "SOFT_MAX 4096x1024", sz = list(rows=1024L, cols=4096L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_soft_max(ctx, a) }),

    list(name = "SOFT_MAX 1x4096",    sz = list(rows=1L, cols=4096L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$cols), as.integer(s$rows))
           ggml_soft_max(ctx, a) }),

    # ------------------------------------------------------------------
    # Transpose / permute / cont
    # ------------------------------------------------------------------
    list(name = "TRANSPOSE 2048x2048", sz = list(R=2048L, C=2048L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$C), as.integer(s$R))
           b <- ggml_transpose(ctx, a)
           ggml_cont(ctx, b) }),

    # ------------------------------------------------------------------
    # CONCAT
    # ------------------------------------------------------------------
    list(name = "CONCAT 2x[2M]", sz = list(N=2e6),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_concat(ctx, a, b, 0L) }),

    # ------------------------------------------------------------------
    # SUM / MEAN / ARGMAX
    # ------------------------------------------------------------------
    list(name = "SUM_ROWS 1024x1024", sz = list(R=1024L, C=1024L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$C), as.integer(s$R))
           ggml_sum_rows(ctx, a) }),

    list(name = "MEAN 1024x1024",    sz = list(R=1024L, C=1024L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$C), as.integer(s$R))
           ggml_mean(ctx, a) }),

    list(name = "ARGMAX 4096",       sz = list(N=4096L, rows=1024L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$N), as.integer(s$rows))
           ggml_argmax(ctx, a) }),

    # ------------------------------------------------------------------
    # REPEAT
    # ------------------------------------------------------------------
    list(name = "REPEAT 1024->4096", sz = list(src=1024L, dst=4096L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$src))
           b <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$dst))
           ggml_repeat(ctx, a, b) }),

    # ------------------------------------------------------------------
    # PAD
    # ------------------------------------------------------------------
    list(name = "PAD 1024x1024",     sz = list(R=1024L, C=1024L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, as.integer(s$C), as.integer(s$R))
           ggml_pad(ctx, a, 8L, 8L) }),

    # ------------------------------------------------------------------
    # UPSCALE
    # ------------------------------------------------------------------
    list(name = "UPSCALE 512x512x3 -> 2x", sz = list(W=512L, H=512L, C=3L),
         flops = NULL,
         build = function(ctx, s) {
           a <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                   as.integer(s$W), as.integer(s$H), as.integer(s$C))
           ggml_upscale(ctx, a, 2L) }),

    # ------------------------------------------------------------------
    # CLAMP
    # ------------------------------------------------------------------
    list(name = "CLAMP 4M",          sz = list(N=4e6),
         flops = function(s) s$N,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N))
           ggml_clamp(ctx, a, -1.0, 1.0) }),

    # ------------------------------------------------------------------
    # GLU variants
    # ------------------------------------------------------------------
    list(name = "GEGLU 2Mx2",        sz = list(N=2e6),
         flops = function(s) s$N * 10,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N) * 2L)
           ggml_geglu(ctx, a) }),

    list(name = "SWIGLU 2Mx2",       sz = list(N=2e6),
         flops = function(s) s$N * 6,
         build = function(ctx, s) {
           a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, as.integer(s$N) * 2L)
           ggml_swiglu(ctx, a) })
  )
}

# ---- system info ------------------------------------------------------------

cat("=== ggmlR Op Benchmark ===\n\n")

nc <- parallel::detectCores(logical = FALSE)
if (is.na(nc)) nc <- 1L
cat(sprintf("CPU threads : %d\n", max(nc - 1L, 1L)))

vulkan_ok <- ggml_vulkan_available()
if (vulkan_ok) {
  gpu_name <- ggml_vulkan_device_description(0L)
  gpu_mem  <- ggml_vulkan_device_memory(0L)
  cat(sprintf("GPU         : %s\n", gpu_name))
  cat(sprintf("VRAM        : %.1f / %.1f GB\n",
              gpu_mem$free / 1e9, gpu_mem$total / 1e9))
} else {
  cat("GPU         : Vulkan not available\n")
}
cat(sprintf("Warmup/Runs : %d / %d\n\n", N_WARMUP, N_RUNS))

# ---- run --------------------------------------------------------------------

ops   <- make_ops()
results <- list()

SEP <- strrep("-", 108)

cat(sprintf("%-32s %9s %9s%-5s %9s %9s %9s %9s\n",
            "Op", "CPU(ms)", "GPU(ms)", "(x)", "min_g", "max_g", "speedup", "GFLOPS_g"))
cat(SEP, "\n")

for (op in ops) {
  cat(sprintf("  %-32s ... ", op$name))

  # capture op vars to avoid closure issues
  .build <- op$build
  .sz    <- op$sz
  .flops <- op$flops

  cpu_r <- bench_op("cpu",    function(ctx) .build(ctx, .sz))
  gpu_r <- if (vulkan_ok) bench_op("vulkan", function(ctx) .build(ctx, .sz)) else NULL

  speedup <- if (!is.null(cpu_r) && !is.null(gpu_r))
               sprintf("%.2fx", cpu_r$mean_ms / gpu_r$mean_ms) else "—"

  gflops_g <- "—"
  if (!is.null(gpu_r) && !is.null(.flops)) {
    gf <- .flops(.sz) / (gpu_r$mean_ms / 1e3) / 1e9
    gflops_g <- sprintf("%.1f", gf)
  }

  r <- list(
    name      = op$name,
    cpu_ms    = if (!is.null(cpu_r)) cpu_r$mean_ms else NA_real_,
    gpu_ms    = if (!is.null(gpu_r)) gpu_r$mean_ms else NA_real_,
    min_g     = if (!is.null(gpu_r)) gpu_r$min_ms  else NA_real_,
    max_g     = if (!is.null(gpu_r)) gpu_r$max_ms  else NA_real_,
    speedup   = speedup,
    gflops_g  = gflops_g,
    batch_cpu = if (!is.null(cpu_r)) cpu_r$n_batch else NA_integer_,
    batch_gpu = if (!is.null(gpu_r)) gpu_r$n_batch else NA_integer_
  )

  cpu_s  <- if (!is.na(r$cpu_ms)) sprintf("%9.3f", r$cpu_ms) else sprintf("%9s", "ERR")
  gpu_s  <- if (!is.na(r$gpu_ms)) sprintf("%9.3f", r$gpu_ms) else sprintf("%9s", if (vulkan_ok) "ERR" else "n/a")
  min_s  <- if (!is.na(r$min_g))  sprintf("%9.3f", r$min_g)  else sprintf("%9s", "—")
  max_s  <- if (!is.na(r$max_g))  sprintf("%9.3f", r$max_g)  else sprintf("%9s", "—")
  nb_s   <- if (!is.na(r$batch_gpu)) sprintf("x%d", r$batch_gpu) else ""
  cat(sprintf("\r%-32s %s %s%s %s %s %9s %9s\n",
              r$name, cpu_s, gpu_s,
              formatC(nb_s, width = 5, flag = "-"),
              min_s, max_s, r$speedup, r$gflops_g))

  results[[length(results) + 1]] <- r
}

cat("\n=== Summary ===\n\n")
cat(sprintf("%-32s %9s %9s %9s\n", "Op", "CPU(ms)", "GPU(ms)", "speedup"))
cat(strrep("-", 62), "\n")
for (r in results) {
  cpu_s <- if (!is.na(r$cpu_ms)) sprintf("%9.3f", r$cpu_ms) else sprintf("%9s", "ERR")
  gpu_s <- if (!is.na(r$gpu_ms)) sprintf("%9.3f", r$gpu_ms) else sprintf("%9s", if (vulkan_ok) "ERR" else "n/a")
  cat(sprintf("%-32s %s %s %9s\n", r$name, cpu_s, gpu_s, r$speedup))
}
cat("\n=== Done ===\n")
