# benchmark_coopmat.R — measure the contribution of GPU Matrix Cores
# (Vulkan cooperative_matrix / coopmat2) to MUL_MAT throughput.
#
# Why a multi-process design:
#   GGML_VK_DISABLE_COOPMAT / GGML_VK_DISABLE_COOPMAT2 are read once, at
#   Vulkan device initialisation (pipeline creation). They cannot be toggled
#   inside a live session. So each configuration runs in its own Rscript
#   child with the env var set, and this parent script collects + compares.
#
# Configurations:
#   full         — coopmat + coopmat2 enabled (default; fastest if supported)
#   no-coopmat2  — GGML_VK_DISABLE_COOPMAT2=1 (falls back to coopmat1)
#   no-coopmat   — GGML_VK_DISABLE_COOPMAT=1 + COOPMAT2=1 (scalar/vec shaders)
#   full-prec32  — as full, with GGML_PREC_F32 on the MUL_MAT node (the f32-tile
#                  kernel ggmlR adds; what matmul_precision = "f32" would select)
#   f32-tiles    — no coopmat and GGML_VK_DISABLE_F16=1: the scalar path with f32
#                  shared memory, the precision reference on the GPU
#
# Each shape also reports relerr = max|C - C_ref| / max|C_ref| against R's %*%
# (shapes up to 2048^3 and the model-sized ones; skipped above that). f16 tiles
# land near 1e-3, f32 near 1e-6.
#
# The full-vs-no-coopmat GFLOPS delta is the measurable Matrix-Core gain.
#
# Usage:
#   Rscript inst/examples/benchmark_coopmat.R           # device 0
#   Rscript inst/examples/benchmark_coopmat.R 0 worker  # internal worker mode

suppressMessages(library(ggmlR))

# ---- shared MUL_MAT benchmark -------------------------------------------------

# Square-ish MUL_MAT shapes spanning small→large. K is the contraction dim.
# FLOPs for C[m,n] += A[m,k]*B[k,n] is 2*m*n*k.
SHAPES <- list(
  c(M = 512L,  N = 512L,  K = 512L),
  c(M = 1024L, N = 1024L, K = 1024L),
  c(M = 2048L, N = 2048L, K = 2048L),
  c(M = 4096L, N = 4096L, K = 4096L),
  c(M = 4096L, N = 4096L, K = 1024L),
  # cayleyR's Q-model: 512 states x 26 tokens through d_model 512 and ff 1024
  c(M = 512L,  N = 13312L, K = 512L),
  c(M = 1024L, N = 13312L, K = 512L)
)
# Defaults with no arguments: 1 warmup, then 2 timed graph computes.
# Override from the environment:
#   BENCH_WARMUP=2 BENCH_RUNS=5 Rscript ... benchmark_coopmat.R
.env_int <- function(name, default) {
  v <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (is.na(v) || v < 1L) default else v
}
N_WARMUP <- .env_int("BENCH_WARMUP", 1L)  # warmup graph computes (untimed)
N_RUNS   <- .env_int("BENCH_RUNS",   2L)  # timed graph computes (averaged)

bench_one <- function(device, shape, prec = "default") {
  M <- shape[["M"]]; N <- shape[["N"]]; K <- shape[["K"]]

  # ggml_mul_mat(a, b): a is [K, M], b is [K, N], result is [M, N].
  # 256 MB matches benchmark_ops.R: with no_alloc=TRUE the context still
  # holds tensor structs + the forward graph (nodes + hash table sized by
  # GGML_DEFAULT_GRAPH_SIZE), which overflows a 64 MB context at 4096^3.
  ctx <- ggml_init(256L * 1024L * 1024L, no_alloc = TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)
  cmat <- ggml_mul_mat(ctx, a, b)
  if (identical(prec, "f32"))
    .Call("R_ggml_mul_mat_set_prec", cmat, 10L, PACKAGE = "ggmlR")  # GGML_PREC_F32
  graph <- ggml_build_forward_expand(ctx, cmat)

  backend <- ggml_vulkan_init(device)
  if (is.null(backend)) stop("Vulkan init failed for device ", device)
  on.exit(ggml_backend_free(backend), add = TRUE)

  buf <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  on.exit(ggml_backend_buffer_free(buf), add = TRUE)

  a_data <- as.numeric(stats::rnorm(K * M))
  b_data <- as.numeric(stats::rnorm(K * N))
  ggml_backend_tensor_set_data(a, a_data)
  ggml_backend_tensor_set_data(b, b_data)

  for (i in seq_len(N_WARMUP)) ggml_backend_graph_compute(backend, graph)

  # Accuracy against R: a is [K, M] and b [K, N] in ggml order, so the result
  # [M, N] is t(A) %*% B with A, B the K-row R matrices.
  relerr <- NA_real_
  if (as.numeric(M) * N * K <= 2048^3 || N == 13312L) {
    ref <- crossprod(matrix(a_data, K, M), matrix(b_data, K, N))
    got <- matrix(ggml_backend_tensor_get_data(cmat), M, N)
    relerr <- max(abs(got - ref)) / max(abs(ref))
  }

  # The user asked for "1 warmup + N_RUNS timed runs". But system.time()
  # resolution is ~10 ms, and every shape except 4096^3 runs far below that,
  # so 2 raw runs round to 0 -> Inf/NA. Instead of dropping those shapes we
  # auto-calibrate: do N_RUNS timed *batches*, where each batch repeats the
  # graph compute enough times to span >= TARGET_S. Per-run time = total /
  # (n_inner * N_RUNS). No explicit synchronize (that hung RADV); the inner
  # loop itself accumulates enough wall time to be measurable.
  TARGET_S <- 0.20  # each timed batch should span at least this long

  probe <- system.time(ggml_backend_graph_compute(backend, graph))[["elapsed"]]
  n_inner <- if (probe >= TARGET_S) 1L else
    max(1L, as.integer(ceiling(TARGET_S / max(probe, 1e-4))))

  per_run <- numeric(N_RUNS)
  for (r in seq_len(N_RUNS)) {
    t <- system.time(
      for (i in seq_len(n_inner)) ggml_backend_graph_compute(backend, graph)
    )[["elapsed"]]
    per_run[r] <- t / n_inner
  }

  mean_per_run <- mean(per_run)
  if (!is.finite(mean_per_run) || mean_per_run <= 0) {
    return(list(ms = NA_real_, gflops = NA_real_, relerr = relerr, untimed = TRUE))
  }
  gflops <- (2 * M * N * K) / mean_per_run / 1e9
  list(ms = mean_per_run * 1000, gflops = gflops, relerr = relerr, untimed = FALSE)
}

run_worker <- function(device, prec = "default") {
  caps <- tryCatch(ggml_vulkan_device_caps(device), error = function(e) NULL)
  cm   <- if (!is.null(caps)) isTRUE(caps$coopmat_support)     else NA
  cmfa <- if (!is.null(caps)) isTRUE(caps$coopmat1_fa_support) else NA
  cmnk <- if (!is.null(caps))
            sprintf("%sx%sx%s", caps$coopmat_m, caps$coopmat_n, caps$coopmat_k)
          else "NA"
  cat(sprintf("CAPS coopmat=%s coopmat1_fa=%s MxNxK=%s subgroup=%s\n",
              cm, cmfa, cmnk, if (!is.null(caps)) caps$subgroup_size else NA))

  for (sh in SHAPES) {
    r <- bench_one(device, sh, prec)
    cat(sprintf("RESULT %dx%dx%d %.3f %.2f %.3g\n",
                sh[["M"]], sh[["N"]], sh[["K"]], r$ms, r$gflops, r$relerr))
  }
}

# ---- parent: spawn one child per configuration --------------------------------

run_parent <- function(device) {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L) {
    stop("No Vulkan device available — coopmat benchmark requires a GPU.")
  }
  cat("Device:", ggml_vulkan_device_description(device), "\n\n")

  # 3rd worker arg encodes which coopmat paths to disable (see entry point).
  # Each entry: the disable flags, then the MUL_MAT precision.
  configs <- list(
    "full"        = c("none", "default"),
    "no-coopmat2" = c("COOPMAT2", "default"),
    "no-coopmat"  = c("COOPMAT2,COOPMAT", "default"),
    "full-prec32" = c("none", "f32"),
    "f32-tiles"   = c("COOPMAT2,COOPMAT,F16", "default")
  )

  self <- normalizePath(sub("^--file=", "",
            grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
  rscript <- file.path(R.home("bin"), "Rscript")

  results <- list()
  for (cfg in names(configs)) {
    cat(sprintf("=== Running configuration: %s ===\n", cfg))
    flag <- configs[[cfg]][[1L]]
    prec <- configs[[cfg]][[2L]]
    # No env= : the child inherits the full parent environment (PATH,
    # VULKAN_SDK, LD_LIBRARY_PATH). The disable flag is a plain arg.
    out <- system2(rscript, c(shQuote(self), device, "worker", flag, prec),
                    stdout = TRUE, stderr = TRUE)
    caps_line <- grep("^CAPS ", out, value = TRUE)
    if (length(caps_line)) cat("  ", caps_line[1], "\n")
    res_lines <- grep("^RESULT ", out, value = TRUE)
    if (!length(res_lines)) {
      cat("  (no results — output below)\n")
      cat(paste0("  | ", out, "\n"))
      next
    }
    parsed <- do.call(rbind, lapply(res_lines, function(l) {
      p <- strsplit(l, " ")[[1]]
      data.frame(shape = p[2], ms = as.numeric(p[3]),
                 gflops = as.numeric(p[4]),
                 relerr = suppressWarnings(as.numeric(p[5])),  # "NA" when skipped
                 stringsAsFactors = FALSE)
    }))
    results[[cfg]] <- parsed
    cat(sprintf("  (%d warmup + %d timed graph computes, averaged)\n",
                N_WARMUP, N_RUNS))
    print(parsed, row.names = FALSE)
    cat("\n")
  }

  # ---- coopmat2 availability note --------------------------------------------
  # coopmat2 == VK_NV_cooperative_matrix2, an NVIDIA-only extension. On AMD /
  # Intel it is never advertised, so GGML_VK_DISABLE_COOPMAT2 is a no-op there
  # and the 'no-coopmat2' run will match 'full'. Detect that empirically (mean
  # GFLOPS within 2% of 'full') and label it instead of leaving it ambiguous.
  if (!is.null(results[["full"]]) && !is.null(results[["no-coopmat2"]])) {
    mf  <- mean(results[["full"]]$gflops)
    mn2 <- mean(results[["no-coopmat2"]]$gflops)
    if (is.finite(mf) && is.finite(mn2) && mf > 0 &&
        abs(mf - mn2) / mf < 0.02) {
      cat("NOTE: 'no-coopmat2' matches 'full' (within 2%) -> coopmat2 is",
          "not active on this device\n")
      cat("      (coopmat2 = VK_NV_cooperative_matrix2, NVIDIA-only;",
          "expected no effect on AMD/Intel).\n\n")
    }
  }

  # ---- summary: which configuration is fastest, and by how much --------------
  # Preserve the SHAPES order (lexical merge would put 512 after 2048).
  shape_order <- vapply(SHAPES, function(s)
    sprintf("%dx%dx%d", s[["M"]], s[["N"]], s[["K"]]), character(1))
  cfg_names <- names(results)

  if (length(cfg_names) >= 2L) {
    cat("=== Summary: GFLOPS per configuration (higher = faster) ===\n")
    tbl <- data.frame(shape = shape_order, stringsAsFactors = FALSE)
    for (cfg in cfg_names) {
      r <- results[[cfg]]
      tbl[[cfg]] <- r$gflops[match(tbl$shape, r$shape)]
    }
    # Per-shape winner and how many times faster than the slowest config.
    fastest  <- character(nrow(tbl))
    slowest  <- character(nrow(tbl))
    ratio    <- numeric(nrow(tbl))
    for (i in seq_len(nrow(tbl))) {
      g <- unlist(tbl[i, cfg_names])
      g <- g[is.finite(g)]
      if (length(g) < 2L) { fastest[i] <- NA; next }
      fastest[i] <- names(which.max(g))
      slowest[i] <- names(which.min(g))
      ratio[i]   <- max(g) / min(g)
    }
    tbl$fastest    <- fastest
    tbl$vs_slowest <- ifelse(is.na(fastest), NA,
                             paste0(round(ratio, 2), "x"))
    show <- tbl
    for (cfg in cfg_names) show[[cfg]] <- round(show[[cfg]], 1)
    print(show, row.names = FALSE)
    cat("\n")

    # Overall verdict: geometric-mean speedup of the fastest config vs the
    # baseline 'no-coopmat' (pure scalar/vec path, no Matrix Cores).
    if (!is.null(results[["full"]]) && !is.null(results[["no-coopmat"]])) {
      f  <- results[["full"]]
      nc <- results[["no-coopmat"]]
      ord <- match(shape_order, f$shape)
      sp  <- f$gflops[ord] /
              nc$gflops[match(shape_order, nc$shape)]
      sp  <- sp[is.finite(sp)]
      if (length(sp)) {
        gm <- exp(mean(log(sp)))
        faster <- gm >= 1
        cat(sprintf(
          "VERDICT: 'full' (coopmat Matrix Cores) is %.2fx %s than 'no-coopmat' (scalar)\n",
          if (faster) gm else 1 / gm,
          if (faster) "FASTER" else "SLOWER"))
        cat(sprintf("         geometric mean over %d shapes; range %.2fx - %.2fx\n",
                    length(sp), min(sp), max(sp)))
      }
    }
  } else {
    cat("Summary skipped: need at least 2 configurations with results.\n")
  }
}

# ---- entry point --------------------------------------------------------------

args   <- commandArgs(trailingOnly = TRUE)
device <- if (length(args) >= 1L) as.integer(args[1]) else 0L
mode   <- if (length(args) >= 2L) args[2] else "parent"

if (identical(mode, "worker")) {
  # Coopmat-disable flags arrive as a 3rd arg ("none" | "COOPMAT2" |
  # "COOPMAT2,COOPMAT"). We Sys.setenv() them HERE, before any Vulkan call,
  # because the C backend reads GGML_VK_DISABLE_COOPMAT* once at device init.
  #
  # Why an arg and not system2(env=): R's system2(env=) *replaces* the child
  # environment, wiping PATH / VULKAN_SDK / LD_LIBRARY_PATH. With those gone
  # the RADV ICD cannot load and the worker hangs (this is exactly why the
  # 'full' run worked but the next one froze). Passing a plain arg lets the
  # child inherit the full parent environment.
  flag <- if (length(args) >= 3L) args[3] else "none"
  if (!identical(flag, "none")) {
    for (v in strsplit(flag, ",")[[1]]) {
      do.call(Sys.setenv, stats::setNames(list("1"),
                                          paste0("GGML_VK_DISABLE_", v)))
    }
  }
  run_worker(device, if (length(args) >= 4L) args[4] else "default")
} else {
  run_parent(device)
}
