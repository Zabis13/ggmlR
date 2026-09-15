#!/usr/bin/env Rscript
# ScatterElements: CPU kernel against Vulkan shader, one node, no model.
#
# This op had three independent defects at the same time and every one of them
# was hunted through a 45 MB detector -- patch the graph, dump nodes, run ONNX
# Runtime, compare, rebuild, minutes per attempt. All three show up in a graph
# of a single node in under a second:
#
#   1. the shader declared a 4-dimension push constant block while the
#      dispatcher sends the 5-dimension vk_op_binary_push_constants, so the
#      axis and the dst strides were read from the wrong offsets
#   2. the pipeline declared a 12-byte push constant range against the 140
#      bytes actually pushed
#   3. the shader indexed only gl_GlobalInvocationID.x while the dispatch
#      spreads work over y once the input passes 65536 elements
#
# ⚠️ The CPU kernel is the reference here, not a second opinion: it is the one
# ONNX Runtime agrees with (max|d| = 7.6e-06 on MaskRCNN-12-int8, against the
# Vulkan path's 7.34). A defect both backends shared would be invisible to
# this script -- for that, ref_check_ops.sh goes through ORT instead.
#
# Usage:
#   Rscript inst/scripts/ref_scatter_cpu_vs_gpu.R

suppressMessages(library(ggmlR))

if (!isTRUE(tryCatch(ggml_vulkan_available(), error = function(e) FALSE))) {
  cat("Vulkan not available -- nothing to compare against\n")
  quit(status = 0)
}

# One scatter, run on one backend, returned as a plain vector.
#
# `targets` gives one destination row per updates row, which is the shape the
# real model uses: a whole row of features moves to a row of a larger output.
run_scatter <- function(backend_name, n_rows, row_len, n_dst, targets,
                        reduction = 0L) {
  stopifnot(length(targets) == n_rows)
  # A target past the end is a bug in the CASE, not a finding about the op, and
  # it surfaces as a length mismatch in the comparison rather than as anything
  # readable. Refuse it here, where the cause is still visible.
  stopifnot(all(targets >= 0L), all(targets < n_dst))

  ctx <- ggml_init(mem_size = 512 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  # ggml is column-major: ne[0] = row_len is the fast axis, ne[1] the row
  # index. Scattering "which row" is therefore axis 1, not 0.
  base <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_len, n_dst)
  upd  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_len, n_rows)
  idx  <- ggml_new_tensor_2d(ctx, GGML_TYPE_I32, row_len, n_rows)
  out  <- ggml_scatter_elements(ctx, base, upd, idx, reduction, 1L)

  backend <- if (backend_name == "vulkan") ggml_vulkan_init(0) else ggml_backend_cpu_init()
  buffer  <- ggml_backend_alloc_ctx_tensors(ctx, backend)

  set.seed(7L)
  upd_vals  <- as.numeric(round(rnorm(n_rows * row_len), 3))
  base_vals <- as.numeric(rep(0, n_dst * row_len))
  # Every element of an updates row carries that row's destination.
  idx_vals  <- as.integer(rep(targets, each = row_len))

  ggml_backend_tensor_set_data(base, base_vals)
  ggml_backend_tensor_set_data(upd,  upd_vals)
  ggml_backend_tensor_set_data(idx,  idx_vals)

  graph <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)
  res <- ggml_backend_tensor_get_data(out)

  ggml_backend_buffer_free(buffer)
  if (backend_name == "vulkan") ggml_vulkan_free(backend) else ggml_backend_free(backend)
  list(res = res, upd = upd_vals)
}

# What the op is defined to produce, computed in R -- so a defect the two
# backends might share still fails the check.
expected <- function(n_rows, row_len, n_dst, targets, upd_vals) {
  out <- rep(0, n_dst * row_len)
  for (r in seq_len(n_rows)) {
    dst <- targets[r]                       # 0-based
    out[(dst * row_len + 1):((dst + 1) * row_len)] <-
      upd_vals[((r - 1) * row_len + 1):(r * row_len)]
  }
  out
}

# Sizes chosen so each one crosses a threshold the others do not.
cases <- list(
  list(name = "tiny",  n_rows = 4L,  row = 3L,    n_dst = 8L,
       targets = c(1L, 5L, 2L, 7L)),
  list(name = "wide",  n_rows = 6L,  row = 512L,  n_dst = 16L,
       targets = c(0L, 15L, 3L, 9L, 1L, 12L)),
  # > 65536 updates: the dispatch splits over y, which is what a shader
  # reading only gl_GlobalInvocationID.x stops covering.
  # ⚠️ Parenthesise the modulo: `(seq_len(40) - 1L) * 3L %% 64L` binds %% to
  # the 3L, so the targets ran to 117 against 64 destination rows and the
  # comparison failed on a length mismatch instead of on the op.
  list(name = "big",   n_rows = 40L, row = 4096L, n_dst = 64L,
       targets = as.integer(((seq_len(40) - 1L) * 3L) %% 64L))
)

cat(sprintf("%-6s %-9s %-10s %-12s %-12s %s\n",
            "case", "updates", "dispatch", "cpu-vs-ref", "gpu-vs-ref", "verdict"))

fails <- 0L
for (cs in cases) {
  n <- cs$n_rows * cs$row
  ne <- ceiling(n / 256)
  disp <- if (ne > 262144) sprintf("{256,256,%d}", ceiling(ne / 65536))
          else if (ne > 256) sprintf("{256,%d,1}", ceiling(ne / 256))
          else sprintf("{%d,1,1}", ne)

  cpu <- run_scatter("cpu",    cs$n_rows, cs$row, cs$n_dst, cs$targets)
  gpu <- run_scatter("vulkan", cs$n_rows, cs$row, cs$n_dst, cs$targets)
  ref <- expected(cs$n_rows, cs$row, cs$n_dst, cs$targets, cpu$upd)

  d_cpu <- max(abs(cpu$res - ref))
  d_gpu <- max(abs(gpu$res - ref))
  ok <- isTRUE(d_cpu <= 1e-5) && isTRUE(d_gpu <= 1e-5)
  if (!ok) fails <- fails + 1L

  cat(sprintf("%-6s %-9d %-10s %-12.4g %-12.4g %s\n",
              cs$name, n, disp, d_cpu, d_gpu, if (ok) "OK" else "FAIL"))

  if (d_gpu > 1e-5) {
    # Which rows landed, not just how far off: a scatter that writes the right
    # values to the wrong rows and one that writes nothing look identical in a
    # max|d|, and they lead to different places.
    nz <- function(v) sum(vapply(seq_len(cs$n_dst), function(i)
      any(v[((i - 1) * cs$row + 1):(i * cs$row)] != 0), logical(1)))
    cat(sprintf("       rows written: gpu=%d cpu=%d expected=%d\n",
                nz(gpu$res), nz(cpu$res), length(unique(cs$targets))))
  }
}

cat(sprintf("\n%d of %d case(s) failed\n", fails, length(cases)))
quit(status = if (fails > 0L) 1L else 0L)
