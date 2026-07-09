#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Isolated MUL_MAT benchmark for P100 — no model, no llamaR.
# Reproduces the hot prefill shapes from the perf log (q6_K / q8_0, m=k=4096, n=488)
# and reports GFLOPS/s. Run twice to A/B the integer-dot (MMQ int8) path:
#
#   Rscript scratch_mulmat_p100_bench.R                                # MMQ int8 ON
#   GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 Rscript scratch_mulmat_p100_bench.R   # OFF (f16 dequant)
#
# Pair it with GGML_VK_PERF_LOGGER=1 to see the per-op GFLOPS from ggml itself.
# ---------------------------------------------------------------------------
suppressMessages(library(ggmlR))

if (!ggml_vulkan_available()) stop("Vulkan GPU not available")

M <- 4096L   # weight rows (output features)
K <- 4096L   # shared dim
N <- 488L    # activation cols (== prefill tokens in the log)
REPS <- 30L

int_dot <- Sys.getenv("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "") == ""
cat(sprintf("=== MUL_MAT bench  m=%d k=%d n=%d  (integer_dot=%s) ===\n",
            M, K, N, if (int_dot) "ON" else "OFF"))

bench <- function(qtype_name, quantize_fn, ggml_type_id) {
  set.seed(1)
  w_raw <- rnorm(M * K)                       # weights, row-major M x K
  w_q   <- quantize_fn(w_raw, n_rows = M, n_per_row = K)
  x_raw <- rnorm(K * N)                       # activations, f32

  # NOTE: the R wrapper's memory pre-check (r_interface.c) miscomputes the size
  # of quantized tensors (multiplies per-block type_size by element count), so it
  # demands ~3.4 GB for a 4096x4096 q6_K tensor. With no_alloc=TRUE nothing is
  # really allocated in the context, so we just size mem_size past that check.
  ctx <- ggml_init(4096 * 1024 * 1024, no_alloc = TRUE)
  a <- ggml_new_tensor_2d(ctx, ggml_type_id, K, M)   # quantized weights (ne0=K, ne1=M)
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N)  # f32 activations
  out <- ggml_mul_mat(ctx, a, b)                     # -> [M, N]

  backend <- ggml_vulkan_init(0)
  ggml_backend_alloc_ctx_tensors(ctx, backend)
  ggml_backend_tensor_set_data(a, w_q)
  ggml_backend_tensor_set_data(b, x_raw)
  gf <- ggml_build_forward_expand(ctx, out)

  ggml_backend_graph_compute(backend, gf)            # warm-up (compile pipeline)
  t0 <- Sys.time()
  for (i in seq_len(REPS)) ggml_backend_graph_compute(backend, gf)
  dt <- as.numeric(Sys.time() - t0, units = "secs") / REPS

  flops  <- 2 * as.numeric(M) * N * K
  gflops <- flops / dt / 1e9
  cat(sprintf("  %-6s : %8.3f ms/run   %8.1f GFLOPS/s\n", qtype_name, dt * 1000, gflops))

  ggml_backend_free(backend); ggml_free(ctx)
}

bench("q6_K", quantize_q6_K, 14L)   # GGML_TYPE_Q6_K = 14
bench("q8_0", quantize_q8_0, 8L)    # GGML_TYPE_Q8_0 = 8
cat("=== done ===\n")
