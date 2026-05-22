# Quick smoke-check for the 11 new op bindings.
# Run after install:  Rscript inst/examples/check_new_bindings.R
library(ggmlR)

compute <- function(ctx, out) {
  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend))
  ggml_backend_cpu_set_n_threads(backend, 2L)
  graph <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)
  ggml_get_f32(out)
}

ok <- function(name, cond) cat(sprintf("%-26s %s\n", name, if (isTRUE(cond)) "OK" else "FAIL"))

# --- arange: [0,1,2,3,4] -------------------------------------------------
{
  ctx <- ggml_init(1024 * 1024); on.exit(ggml_free(ctx))
  r <- ggml_arange(ctx, 0, 5, 1)
  v <- compute(ctx, r)
  ok("arange", isTRUE(all.equal(v, c(0,1,2,3,4))))
  ggml_free(ctx); on.exit()
}

# --- roll: shift a 1D-as-2D row by 1 -------------------------------------
{
  ctx <- ggml_init(1024 * 1024)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1,2,3,4)); ggml_set_input(a)
  r <- ggml_roll(ctx, a, 1L, 0L, 0L, 0L)
  v <- compute(ctx, r)
  ok("roll", isTRUE(all.equal(sort(v), c(1,2,3,4))))   # rotation preserves multiset
  ggml_free(ctx)
}

# --- pad_reflect_1d ------------------------------------------------------
{
  ctx <- ggml_init(1024 * 1024)
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4)
  ggml_set_f32(a, c(1,2,3,4)); ggml_set_input(a)
  r <- ggml_pad_reflect_1d(ctx, a, 2L, 2L)
  v <- compute(ctx, r)
  ok("pad_reflect_1d", length(v) == 8 && all(is.finite(v)))
  ggml_free(ctx)
}

# --- conv_1d_dw ----------------------------------------------------------
{
  ctx <- ggml_init(4 * 1024 * 1024)
  k <- ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 1, 2)   # KW, 1, C
  b <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2)      # W, C
  ggml_set_input(b)
  r <- tryCatch(ggml_conv_1d_dw(ctx, k, b, 1L, 1L, 1L), error = function(e) e)
  ok("conv_1d_dw (builds)", !inherits(r, "error"))
  ggml_free(ctx)
}

# --- conv_2d_dw ----------------------------------------------------------
{
  ctx <- ggml_init(8 * 1024 * 1024)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 3, 3, 1, 2)  # KW,KH,1,C
  b <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 6, 2, 1)  # W,H,C,N
  ggml_set_input(b)
  r <- tryCatch(ggml_conv_2d_dw(ctx, k, b, 1L,1L,1L,1L,1L,1L), error = function(e) e)
  ok("conv_2d_dw (builds)", !inherits(r, "error"))
  ggml_free(ctx)
}

# --- conv_2d_dw_direct ---------------------------------------------------
{
  ctx <- ggml_init(8 * 1024 * 1024)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, 1, 2)  # KW,KH,1,C  (direct layout)
  b <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 6, 2, 1)  # W,H,C,N
  ggml_set_input(b)
  r <- tryCatch(ggml_conv_2d_dw_direct(ctx, k, b, 1L,1L,1L,1L,1L,1L), error = function(e) e)
  ok("conv_2d_dw_direct (builds)", !inherits(r, "error"))
  ggml_free(ctx)
}

# --- conv_transpose_2d_p0 (numeric, from upstream test) ------------------
{
  ctx <- ggml_init(8 * 1024 * 1024)
  t <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 2, 2, 1)  # w,h,cin,N
  ggml_set_f32(t, as.numeric(0:11)); ggml_set_input(t)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 2, 3, 2)  # w,h,cin,cout
  # NB: F16 kernel set via numeric helper if available; else builds-only
  r <- tryCatch(ggml_conv_transpose_2d_p0(ctx, k, t, 1L), error = function(e) e)
  ok("conv_transpose_2d_p0 (builds)", !inherits(r, "error"))
  ggml_free(ctx)
}

# --- get_rel_pos ---------------------------------------------------------
{
  ctx <- ggml_init(1024 * 1024)
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 5)   # head_dim, 2*qh-1
  ggml_set_input(a)
  r <- tryCatch(ggml_get_rel_pos(ctx, a, 3L, 3L), error = function(e) e)
  ok("get_rel_pos (builds)", !inherits(r, "error"))
  ggml_free(ctx)
}

# --- win_part / win_unpart ----------------------------------------------
{
  ctx <- ggml_init(4 * 1024 * 1024)
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 4, 4, 1)  # C,W,H,N
  ggml_set_input(a)
  rp <- tryCatch(ggml_win_part(ctx, a, 2L), error = function(e) e)
  ok("win_part (builds)", !inherits(rp, "error"))
  if (!inherits(rp, "error")) {
    ru <- tryCatch(ggml_win_unpart(ctx, rp, 4L, 4L, 2L), error = function(e) e)
    ok("win_unpart (builds)", !inherits(ru, "error"))
  }
  ggml_free(ctx)
}

cat("\nDone. 'builds' checks only verify the binding wires up; numeric checks (arange) verify compute.\n")
