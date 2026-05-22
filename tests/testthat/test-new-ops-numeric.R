# Numeric tests for the op bindings added to mirror upstream ggml tests:
# arange, roll, pad_reflect_1d (cf. test-backend-ops.cpp) and
# get_rel_pos / add_rel_pos / win_part / win_unpart (cf. test-rel-pos.c).
#
# These ops previously had bindings but no value verification.

library(ggmlR)

compute_f32 <- function(ctx, out) {
  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be))
  ggml_backend_cpu_set_n_threads(be, 2L)
  g <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(be, g)
  ggml_get_f32(out)
}

# ---- arange --------------------------------------------------------------
test_that("arange produces [start, stop) with step", {
  ctx <- ggml_init(1024 * 1024); on.exit(ggml_free(ctx))
  expect_equal(compute_f32(ctx, ggml_arange(ctx, 0, 5, 1)), c(0, 1, 2, 3, 4))
})

test_that("arange honours non-unit step", {
  ctx <- ggml_init(1024 * 1024); on.exit(ggml_free(ctx))
  expect_equal(compute_f32(ctx, ggml_arange(ctx, 2, 10, 2)), c(2, 4, 6, 8))
})

# ---- roll ----------------------------------------------------------------
test_that("roll circularly shifts along dim 0", {
  ctx <- ggml_init(1024 * 1024); on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5)); ggml_set_input(a)
  expect_equal(compute_f32(ctx, ggml_roll(ctx, a, 1L, 0L, 0L, 0L)),
               c(5, 1, 2, 3, 4))               # shift +1
})

test_that("roll handles negative shift", {
  ctx <- ggml_init(1024 * 1024); on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5)
  ggml_set_f32(a, c(1, 2, 3, 4, 5)); ggml_set_input(a)
  expect_equal(compute_f32(ctx, ggml_roll(ctx, a, -1L, 0L, 0L, 0L)),
               c(2, 3, 4, 5, 1))               # shift -1
})

# ---- pad_reflect_1d ------------------------------------------------------
test_that("pad_reflect_1d reflects without repeating the edge", {
  ctx <- ggml_init(1024 * 1024); on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1)
  ggml_set_f32(a, c(1, 2, 3, 4)); ggml_set_input(a)
  # [1 2 3 4] padded 2 left / 2 right -> 3 2 |1 2 3 4| 3 2
  expect_equal(compute_f32(ctx, ggml_pad_reflect_1d(ctx, a, 2L, 2L)),
               c(3, 2, 1, 2, 3, 4, 3, 2))
})

# ---- win_part / win_unpart (round trip) ----------------------------------
test_that("win_part then win_unpart reconstructs the input", {
  ctx <- ggml_init(8 * 1024 * 1024); on.exit(ggml_free(ctx))
  a <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, 4, 4, 1)  # C, W, H, N
  vals <- as.numeric(1:32)
  ggml_set_f32(a, vals); ggml_set_input(a)

  wp <- ggml_win_part(ctx, a, 2L)
  expect_equal(ggml_tensor_shape(wp), c(2, 2, 2, 4))       # 4 windows of 2x2

  wu <- ggml_win_unpart(ctx, wp, 4L, 4L, 2L)
  expect_equal(ggml_tensor_shape(wu), c(2, 4, 4, 1))
  expect_equal(compute_f32(ctx, wu), vals)
})

# ---- get_rel_pos + add_rel_pos (ported from test-rel-pos.c) --------------
test_that("get_rel_pos + add_rel_pos match upstream reference values", {
  ctx <- ggml_init(8 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  t  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 3, 3)
  t2 <- ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 3, 3)
  rw  <- ggml_get_rel_pos(ctx, t,  2L, 2L)                  # -> [3,2,2]
  rh  <- ggml_get_rel_pos(ctx, t2, 2L, 2L)
  rwf <- ggml_cpy(ctx, rw, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2))
  rhf <- ggml_cpy(ctx, rh, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2))
  inp <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 9, 4)
  out <- ggml_add_rel_pos(ctx, inp, rwf, rhf)

  be <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(be), add = TRUE)
  ggml_backend_cpu_set_n_threads(be, 2L)
  ggml_backend_alloc_ctx_tensors(ctx, be)

  ggml_backend_tensor_set_data(t,  as.numeric(0:8))        # buf_f16 from 0
  ggml_backend_tensor_set_data(t2, as.numeric(1:9))        # buf_f16 + 1
  ggml_backend_tensor_set_data(inp, rep(1, 36))

  gf <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(be, gf)
  v <- ggml_backend_tensor_get_data(out)

  # upstream expected_out (4 rows x 9), laid out column-wise here (ne[0]=9)
  expected <- c(
    8, 9, 10, 9, 10, 11, 10, 11, 12,
    2, 3,  4, 3,  4,  5,  4,  5,  6,
    14, 15, 16, 15, 16, 17, 16, 17, 18,
    8, 9, 10, 9, 10, 11, 10, 11, 12)
  expect_equal(v, expected, tolerance = 1e-4)
})
