#!/usr/bin/env Rscript
# Benchmark: conv2d scalar vs cm1, F32 vs F16 kernel
# Run: Rscript inst/examples/bench_conv2d_cm1.R

library(ggmlR)

cat("=== conv2d scalar vs cm1 benchmark ===\n\n")

if (!ggml_vulkan_available()) stop("Vulkan not compiled")

caps <- ggml_vulkan_device_caps(0L)
cat(sprintf("Device        : %s\n", ggml_vulkan_device_description(0L)))
cat(sprintf("coopmat       : %s  subgroup_size=%d\n",
            if (caps$coopmat_support) "YES" else "NO", caps$subgroup_size))
cat("\n")

if (!caps$coopmat_support) stop("No coopmat — cm1 unavailable")

# ---------------------------------------------------------------------------
# Subprocess runner: returns ms vector for given (disable_coopmat, f16_kernel)
# ---------------------------------------------------------------------------
run_bench_subprocess <- function(disable_coopmat, f16_kernel) {
  script <- tempfile(fileext = ".R")
  out    <- tempfile(fileext = ".rds")

  shapes <- list(
    list(N=1L, Cin=512L, Cout=512L, H=32L,  W=32L),
    list(N=1L, Cin=256L, Cout=512L, H=64L,  W=64L),
    list(N=1L, Cin=128L, Cout=256L, H=128L, W=128L),
    list(N=1L, Cin=64L,  Cout=128L, H=256L, W=256L),
    list(N=1L, Cin=32L,  Cout=64L,  H=256L, W=256L)
  )

  knl_type_str <- if (f16_kernel) "GGML_TYPE_F16" else "GGML_TYPE_F32"

  writeLines(c(
    "library(ggmlR)",
    "ggml_backend_load_all()",
    "gpu <- ggml_backend_init_best()",
    sprintf("knl_type <- %s", knl_type_str),
    "run_conv2d <- function(backend, knl_data, src_data,",
    "                       Cout, Cin, KH, KW, N, H, W, ktype,",
    "                       stride = 1L, pad = 1L) {",
    "  mem  <- as.numeric(Cout * Cin * KH * KW + W * H * Cin * N) * 4 * 16",
    "  ctx  <- ggml_init(mem_size = max(mem, 64 * 1024 * 1024))",
    "  ggml_set_no_alloc(ctx, TRUE)",
    "  knl  <- ggml_new_tensor_4d(ctx, ktype, KW, KH, Cin, Cout)",
    "  src  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, N)",
    "  dst  <- ggml_conv_2d(ctx, knl, src,",
    "                       as.integer(stride), as.integer(stride),",
    "                       as.integer(pad), as.integer(pad), 1L, 1L)",
    "  graph <- ggml_build_forward_expand(ctx, dst)",
    "  buf   <- ggml_backend_alloc_ctx_tensors(ctx, backend)",
    "  ggml_backend_tensor_set_data(knl, knl_data)",  # auto-converts to F16 if ktype==F16
    "  ggml_backend_tensor_set_data(src, src_data)",
    "  ggml_backend_graph_compute(backend, graph)",
    "  out <- ggml_backend_tensor_get_data(dst)",
    "  ggml_backend_buffer_free(buf)",
    "  ggml_free(ctx)",
    "  out",
    "}",
    "bench_shape <- function(N, Cin, Cout, H, W, KH=3L, KW=3L,",
    "                        stride=1L, pad=1L, reps=5L) {",
    "  set.seed(1L)",
    "  knl_d <- rnorm(KW*KH*Cin*Cout, sd=0.02)",
    "  src_d <- rnorm(W*H*Cin*N,      sd=0.02)",
    "  for (i in seq_len(3L))",
    "    run_conv2d(gpu, knl_d, src_d, Cout, Cin, KH, KW, N, H, W, knl_type, stride, pad)",
    "  times <- numeric(reps)",
    "  for (i in seq_len(reps)) {",
    "    t0 <- proc.time()[[\"elapsed\"]]",
    "    run_conv2d(gpu, knl_d, src_d, Cout, Cin, KH, KW, N, H, W, knl_type, stride, pad)",
    "    times[i] <- (proc.time()[[\"elapsed\"]] - t0) * 1e3",
    "  }",
    "  median(times)",
    "}",
    sprintf("shapes <- %s", deparse(shapes)),
    sprintf("out_file <- %s", deparse(out)),
    "results <- sapply(shapes, function(s)",
    "  tryCatch(bench_shape(s$N, s$Cin, s$Cout, s$H, s$W), error=function(e) NA_real_))",
    "saveRDS(results, out_file)"
  ), script)

  env <- if (disable_coopmat) "GGML_VK_DISABLE_COOPMAT=1" else ""
  ret <- system(sprintf("%s Rscript --vanilla %s 2>/dev/null", env, script),
                ignore.stdout = FALSE)
  if (ret != 0 || !file.exists(out)) {
    unlink(script)
    return(rep(NA_real_, length(shapes)))
  }
  res <- readRDS(out)
  unlink(c(script, out))
  res
}

shapes <- list(
  list(N=1L, Cin=512L, Cout=512L, H=32L,  W=32L),
  list(N=1L, Cin=256L, Cout=512L, H=64L,  W=64L),
  list(N=1L, Cin=128L, Cout=256L, H=128L, W=128L),
  list(N=1L, Cin=64L,  Cout=128L, H=256L, W=256L),
  list(N=1L, Cin=32L,  Cout=64L,  H=256L, W=256L)
)
labels <- sapply(shapes, function(s)
  sprintf("%d<-%d @ %dx%d k3", s$Cout, s$Cin, s$H, s$W))

# ---------------------------------------------------------------------------
# Correctness: F16 kernel cm1 vs CPU F32
# ---------------------------------------------------------------------------
cat("--- Correctness (F16-kernel GPU cm1 vs CPU F32) ---\n")
{
  gpu <- ggml_backend_init_best()
  cpu <- ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu, 1L)

  run_one <- function(backend, knl_data, src_data, Cout, Cin, KH, KW, N, H, W, ktype) {
    mem  <- as.numeric(Cout * Cin * KH * KW + W * H * Cin * N) * 4 * 16
    ctx  <- ggml_init(mem_size = max(mem, 64 * 1024 * 1024))
    ggml_set_no_alloc(ctx, TRUE)
    knl  <- ggml_new_tensor_4d(ctx, ktype,          KW, KH, Cin, Cout)
    src  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  W,  H,  Cin, N)
    dst  <- ggml_conv_2d(ctx, knl, src, 1L, 1L, 1L, 1L, 1L, 1L)
    graph <- ggml_build_forward_expand(ctx, dst)
    buf   <- ggml_backend_alloc_ctx_tensors(ctx, backend)
    ggml_backend_tensor_set_data(knl, knl_data)  # auto-converts to F16 if ktype==F16
    ggml_backend_tensor_set_data(src, src_data)
    ggml_backend_graph_compute(backend, graph)
    out <- ggml_backend_tensor_get_data(dst)
    ggml_backend_buffer_free(buf); ggml_free(ctx)
    out
  }

  set.seed(42L)
  kd <- rnorm(3*3*8*16, sd=0.05); sd_ <- rnorm(16*16*8, sd=0.05)
  g16 <- run_one(gpu, kd, sd_, 16L, 8L, 3L, 3L, 1L, 16L, 16L, GGML_TYPE_F16)
  cpu_ <- run_one(cpu, kd, sd_, 16L, 8L, 3L, 3L, 1L, 16L, 16L, GGML_TYPE_F32)
  err <- max(abs(g16 - cpu_))
  cat(sprintf("  max |F16-GPU - F32-CPU| = %.3e  %s\n", err,
              if (err < 5e-3) "PASS" else "FAIL"))
}

# ---------------------------------------------------------------------------
# Timing: 4 variants
# ---------------------------------------------------------------------------
variants <- list(
  list(label="scalar F32", disable_coopmat=TRUE,  f16=FALSE),
  list(label="scalar F16", disable_coopmat=TRUE,  f16=TRUE),
  list(label="cm1    F32", disable_coopmat=FALSE, f16=FALSE),
  list(label="cm1    F16", disable_coopmat=FALSE, f16=TRUE)
)

cat("\n--- Timing (median of 5 reps) ---\n")
hdr <- sprintf("  %-28s", "Cout<-Cin @ HxW k3")
for (v in variants) hdr <- paste0(hdr, sprintf("  %10s", v$label))
cat(hdr, "\n")
cat(sprintf("  %s\n", strrep("-", 28 + 4*12)))

results <- list()
for (v in variants) {
  cat(sprintf("  [running %s...]\n", v$label))
  results[[v$label]] <- run_bench_subprocess(v$disable_coopmat, v$f16)
}

for (i in seq_along(shapes)) {
  row <- sprintf("  %-28s", labels[i])
  best <- min(sapply(results, function(r) r[i]), na.rm=TRUE)
  for (v in variants) {
    ms <- results[[v$label]][i]
    marker <- if (!is.na(ms) && ms == best) "*" else " "
    row <- paste0(row, sprintf(" %8.1f%s ", ms, marker))
  }
  cat(row, "\n")
}
cat("  (* = fastest)\n\nDone.\n")
