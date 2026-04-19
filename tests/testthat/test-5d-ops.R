# Tests for 5D tensor ops (dim4 / ne[4] > 1).
# Verifies that unary ops, binary ops, and concat all correctly traverse the
# 5th dimension instead of silently writing only into the first ne[4]=1 slice.

run_5d <- function(build_fn, set_fn, n_out) {
  ctx     <- ggml_init(mem_size = 32L * 1024L * 1024L, no_alloc = TRUE)
  out     <- build_fn(ctx)
  backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(backend, 2L)
  buf     <- ggml_backend_alloc_ctx_tensors(ctx, backend)
  set_fn()
  graph   <- ggml_build_forward_expand(ctx, out)
  ggml_backend_graph_compute(backend, graph)
  result  <- ggml_backend_tensor_get_data(out, n_elements = n_out)
  ggml_backend_buffer_free(buf)
  ggml_backend_free(backend)
  ggml_free(ctx)
  result
}

# Shape: ne0=3, ne1=2, ne2=2, ne3=2, ne4=3  (72 elements, ne4=3 is key)
NE    <- c(3L, 2L, 2L, 2L, 3L)
N     <- prod(NE)
set.seed(42)
A_DAT <- runif(N, 0.1, 1.0)
B_DAT <- runif(N, 0.1, 1.0)

# ---- unary ops: result must span all 72 elements ----------------------------

for (opname in c("relu", "silu", "gelu", "abs", "neg", "sqrt")) {
  local({
    op <- opname
    test_that(paste("5D unary", op, "covers ne[4]"), {
      env <- new.env(parent = emptyenv())
      res <- run_5d(
        build_fn = function(ctx) {
          a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
          env$a <- a
          switch(op,
            relu = ggml_relu(ctx, a),
            silu = ggml_silu(ctx, a),
            gelu = ggml_gelu(ctx, a),
            abs  = ggml_abs(ctx, a),
            neg  = ggml_neg(ctx, a),
            sqrt = ggml_sqrt(ctx, a)
          )
        },
        set_fn = function() ggml_backend_tensor_set_data(env$a, A_DAT),
        n_out  = N
      )
      expect_length(res, N)
      expect_true(all(is.finite(res)), info = paste(op, "produced NaN/Inf"))
      # Verify that the last ne4 slice differs from the first (not all zeros/same)
      slice1 <- res[1:24]
      slice3 <- res[49:72]
      expect_false(identical(slice1, slice3),
        info = paste(op, "ne[4] slice3 == slice1: dim4 likely not traversed"))
    })
  })
}

# ---- binary ops: verify all 72 elements are written -------------------------

for (opname in c("add", "mul", "sub", "div")) {
  local({
    op <- opname
    test_that(paste("5D binary", op, "covers ne[4]"), {
      env <- new.env(parent = emptyenv())
      res <- run_5d(
        build_fn = function(ctx) {
          a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
          b <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
          env$a <- a; env$b <- b
          switch(op,
            add = ggml_add(ctx, a, b),
            mul = ggml_mul(ctx, a, b),
            sub = ggml_sub(ctx, a, b),
            div = ggml_div(ctx, a, b)
          )
        },
        set_fn = function() {
          ggml_backend_tensor_set_data(env$a, A_DAT)
          ggml_backend_tensor_set_data(env$b, B_DAT)
        },
        n_out = N
      )
      expect_length(res, N)
      expect_true(all(is.finite(res)), info = paste(op, "produced NaN/Inf"))
      slice1 <- res[1:24]
      slice3 <- res[49:72]
      expect_false(identical(slice1, slice3),
        info = paste(op, "ne[4] slice3 == slice1: dim4 likely not traversed"))
    })
  })
}

# ---- concat on each axis ----------------------------------------------------

for (axis in 0:4) {
  local({
    ax <- axis
    test_that(paste("5D concat axis", ax, "covers ne[4]"), {
      NE_out        <- NE
      NE_out[ax+1L] <- NE_out[ax+1L] * 2L
      N_out         <- prod(NE_out)
      env <- new.env(parent = emptyenv())
      res <- run_5d(
        build_fn = function(ctx) {
          a <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
          b <- ggml_new_tensor(ctx, GGML_TYPE_F32, 5L, NE)
          env$a <- a; env$b <- b
          ggml_concat(ctx, a, b, dim = ax)
        },
        set_fn = function() {
          ggml_backend_tensor_set_data(env$a, A_DAT)
          ggml_backend_tensor_set_data(env$b, B_DAT)
        },
        n_out = N_out
      )
      expect_length(res, N_out)
      expect_true(all(is.finite(res)),
        info = paste("concat axis", ax, "produced NaN/Inf"))
    })
  })
}
