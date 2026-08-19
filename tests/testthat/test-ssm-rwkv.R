# Tests for the state-space (Mamba) and RWKV-family recurrence bindings.
#
# These ops share an unusual convention: each returns ONE tensor with the
# sequence output and the final recurrent state concatenated. Most of what can
# go wrong is in the shapes and in the offset arithmetic of the view helpers, so
# that is what these check -- plus a reference computation for ssm_conv, whose
# math is a plain depthwise convolution and so is verifiable in R.
#
# ssm_conv is differentiable (ggmlR adds the backward pass upstream lacks), and
# the gradient checks for it live at the end of this file. The other four --
# ssm_scan, rwkv_wkv6/7 and gated_linear_attn -- are still inference-only, so
# for those there is nothing to train and no gradient to check.

# Build a forward graph for `out` and compute it on the CPU.
compute_1 <- function(ctx, out) {
  graph <- ggml_build_forward_expand(ctx, out)
  ggml_graph_compute(ctx, graph)
  invisible(NULL)
}

# ---------------------------------------------------------------------------
# ssm_conv
# ---------------------------------------------------------------------------

test_that("ggml_ssm_conv has the documented output shape", {
  ctx <- ggml_init(16 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  d_conv <- 4L; d_inner <- 3L; n_t <- 5L; n_s <- 2L
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_conv - 1L + n_t, d_inner, n_s)
  cc <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
  ggml_set_f32(sx, rep(0, ggml_nelements(sx)))
  ggml_set_f32(cc, rep(0, ggml_nelements(cc)))

  out <- ggml_ssm_conv(ctx, sx, cc)
  ne  <- ggml_tensor_shape(out)
  expect_equal(ne[1], d_inner)
  expect_equal(ne[2], n_t)
  expect_equal(ne[3], n_s)
})

test_that("ggml_ssm_conv matches a depthwise convolution computed in R", {
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  d_conv <- 3L; d_inner <- 2L; n_t <- 4L; n_s <- 1L
  len <- d_conv - 1L + n_t

  set.seed(11L)
  sx_v <- runif(len * d_inner * n_s, -1, 1)
  c_v  <- runif(d_conv * d_inner, -1, 1)

  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, len, d_inner, n_s)
  cc <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
  ggml_set_f32(sx, sx_v)
  ggml_set_f32(cc, c_v)

  out <- ggml_ssm_conv(ctx, sx, cc)
  compute_1(ctx, out)
  got <- ggml_get_f32(out)

  # Reference: each channel is convolved with its own kernel, and output
  # position t sums the d_conv taps ending at t. Column-major throughout:
  # sx[i, ch] is sx_v[(ch - 1) * len + i], c[j, ch] is c_v[(ch - 1) * d_conv + j].
  ref <- numeric(d_inner * n_t)
  for (ch in seq_len(d_inner)) {
    for (t in seq_len(n_t)) {
      acc <- 0
      for (j in seq_len(d_conv)) {
        acc <- acc + sx_v[(ch - 1L) * len + (t + j - 1L)] *
                     c_v[(ch - 1L) * d_conv + j]
      }
      # result is [d_inner, n_t], so channel is the fastest axis
      ref[(t - 1L) * d_inner + ch] <- acc
    }
  }

  expect_equal(got, ref, tolerance = 1e-5)
})

# ---------------------------------------------------------------------------
# ssm_scan -- shapes and the packed-result helpers
# ---------------------------------------------------------------------------

# One consistent set of ssm_scan operands, built to the contract in the header.
make_scan <- function(ctx, d_state = 4L, head_dim = 2L, n_head = 3L,
                      n_seq_tokens = 2L, n_seqs = 2L, n_group = 1L,
                      fill = 0) {
  s   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs)
  x   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs)
  dt  <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs)
  A   <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_head)      # Mamba-2 form
  B   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs)
  Cc  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs)
  ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs)

  for (t in list(s, x, dt, A, B, Cc)) ggml_set_f32(t, rep(fill, ggml_nelements(t)))
  ggml_set_i32(ids, seq_len(n_seqs) - 1L)      # 0-based, one state row each

  list(s = s, x = x, dt = dt, A = A, B = B, C = Cc, ids = ids,
       d_state = d_state, head_dim = head_dim, n_head = n_head,
       n_seq_tokens = n_seq_tokens, n_seqs = n_seqs)
}

test_that("ggml_ssm_scan packs outputs and states into one flat tensor", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_scan(ctx)
  r <- ggml_ssm_scan(ctx, p$s, p$x, p$dt, p$A, p$B, p$C, p$ids)

  # nelements(x) outputs, then d_state*head_dim*n_head*n_seqs states.
  expect_equal(ggml_nelements(r),
               ggml_nelements(p$x) +
                 p$d_state * p$head_dim * p$n_head * p$n_seqs)
})

test_that("ggml_ssm_scan_output views the outputs with x's shape", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_scan(ctx)
  r <- ggml_ssm_scan(ctx, p$s, p$x, p$dt, p$A, p$B, p$C, p$ids)
  y <- ggml_ssm_scan_output(ctx, r, p$x)

  expect_equal(ggml_tensor_shape(y), ggml_tensor_shape(p$x))
  expect_equal(ggml_nelements(y), ggml_nelements(p$x))
})

test_that("ggml_ssm_scan_state views the trailing states", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_scan(ctx)
  r <- ggml_ssm_scan(ctx, p$s, p$x, p$dt, p$A, p$B, p$C, p$ids)
  st <- ggml_ssm_scan_state(ctx, r, p$s, p$ids)

  ne <- ggml_tensor_shape(st)
  expect_equal(ne[1], p$d_state)
  expect_equal(ne[2], p$head_dim)
  expect_equal(ne[3], p$n_head)
  expect_equal(ne[4], p$n_seqs)
  # Together the two views cover the packed result exactly -- no gap, no overlap.
  y <- ggml_ssm_scan_output(ctx, r, p$x)
  expect_equal(ggml_nelements(y) + ggml_nelements(st), ggml_nelements(r))
})

test_that("the ssm_scan views read back the halves they point at", {
  # Zero inputs make the outputs zero whatever the recurrence does, so any
  # value the output view picks up would mean it is reading the state block.
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_scan(ctx, fill = 0)
  r <- ggml_ssm_scan(ctx, p$s, p$x, p$dt, p$A, p$B, p$C, p$ids)
  y <- ggml_ssm_scan_output(ctx, r, p$x)
  compute_1(ctx, r)

  vals <- ggml_get_f32(y)
  expect_length(vals, ggml_nelements(p$x))
  expect_true(all(is.finite(vals)))
  expect_equal(vals, rep(0, length(vals)), tolerance = 1e-6)
})

# ---------------------------------------------------------------------------
# RWKV-6 / RWKV-7 / GLA -- shapes and the shared packed-result helpers
# ---------------------------------------------------------------------------

# k, v and friends are all [S, H, n_tokens]; the state holds S*S*H*n_seqs.
#
# n_tokens counts the tokens of ALL sequences together, so it must be a
# multiple of n_seqs: the kernel derives the per-sequence length as
# T / n_seqs and starts a fresh state whenever t % (T / n_seqs) == 0. An
# indivisible pair (3 tokens over 2 sequences) reads past the state buffer and
# segfaults rather than erroring, so keep these two in step.
# H is 4 rather than 2 on purpose: the kernel lets threads with ith >= HEADS
# return BEFORE its ggml_barrier(), so running it with more threads than heads
# deadlocks. Tests run on 2 threads, but a head count above that keeps this
# from depending on the machine.
make_rwkv <- function(ctx, S = 4L, H = 4L, n_tokens = 4L, n_seqs = 2L,
                      fill = 0) {
  stopifnot(n_tokens %% n_seqs == 0L)
  mk3 <- function() {
    t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens)
    ggml_set_f32(t, rep(fill, ggml_nelements(t)))
    t
  }
  state <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S * H, n_seqs)
  ggml_set_f32(state, rep(fill, ggml_nelements(state)))
  tf <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, H)
  ggml_set_f32(tf, rep(fill, ggml_nelements(tf)))

  list(S = S, H = H, n_tokens = n_tokens, n_seqs = n_seqs,
       state = state, tf = tf,
       k = mk3(), v = mk3(), r = mk3(), td = mk3(),
       w = mk3(), a = mk3(), b = mk3(), q = mk3(), g = mk3())
}

test_that("ggml_rwkv_wkv6 has the documented packed shape", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_rwkv(ctx)
  out <- ggml_rwkv_wkv6(ctx, p$k, p$v, p$r, p$tf, p$td, p$state)
  ne  <- ggml_tensor_shape(out)

  expect_equal(ne[1], p$S * p$H)
  expect_equal(ne[2], p$n_tokens + p$S * p$n_seqs)
})

test_that("ggml_rwkv_wkv7 has the documented packed shape", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_rwkv(ctx)
  out <- ggml_rwkv_wkv7(ctx, p$r, p$w, p$k, p$v, p$a, p$b, p$state)
  ne  <- ggml_tensor_shape(out)

  expect_equal(ne[1], p$S * p$H)
  expect_equal(ne[2], p$n_tokens + p$S * p$n_seqs)
})

test_that("ggml_gated_linear_attn has the documented packed shape", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_rwkv(ctx)
  out <- ggml_gated_linear_attn(ctx, p$k, p$v, p$q, p$g, p$state,
                                scale = 1 / sqrt(p$S))
  ne  <- ggml_tensor_shape(out)

  expect_equal(ne[1], p$S * p$H)
  expect_equal(ne[2], p$n_tokens + p$S * p$n_seqs)
})

test_that("the RWKV views split the packed result exactly", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p   <- make_rwkv(ctx)
  out <- ggml_rwkv_wkv6(ctx, p$k, p$v, p$r, p$tf, p$td, p$state)
  y   <- ggml_rwkv_output(ctx, out, p$k)
  st  <- ggml_rwkv_state(ctx, out, p$k, p$state)

  expect_equal(ggml_tensor_shape(y)[1], p$S * p$H)
  expect_equal(ggml_tensor_shape(y)[2], p$n_tokens)
  expect_equal(ggml_tensor_shape(st)[1], p$S * p$H)
  expect_equal(ggml_tensor_shape(st)[2], p$S * p$n_seqs)
  # No gap and no overlap between the two halves.
  expect_equal(ggml_nelements(y) + ggml_nelements(st), ggml_nelements(out))
})

test_that("the RWKV views work for wkv7 and GLA too", {
  # The three ops share one output layout, so one pair of helpers covers them.
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_rwkv(ctx)
  for (out in list(
        ggml_rwkv_wkv7(ctx, p$r, p$w, p$k, p$v, p$a, p$b, p$state),
        ggml_gated_linear_attn(ctx, p$k, p$v, p$q, p$g, p$state, 1 / sqrt(p$S)))) {
    y  <- ggml_rwkv_output(ctx, out, p$k)
    st <- ggml_rwkv_state(ctx, out, p$k, p$state)
    expect_equal(ggml_tensor_shape(y)[2], p$n_tokens)
    expect_equal(ggml_tensor_shape(st)[2], p$S * p$n_seqs)
    expect_equal(ggml_nelements(y) + ggml_nelements(st), ggml_nelements(out))
  }
})

test_that("rwkv_wkv6 computes and its output view reads back finite values", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  set.seed(3L)
  p <- make_rwkv(ctx, fill = 0)
  # A decaying, non-degenerate set of operands: all-zeros would pass even if
  # the op never ran.
  ggml_set_f32(p$k,  runif(ggml_nelements(p$k),  -0.5, 0.5))
  ggml_set_f32(p$v,  runif(ggml_nelements(p$v),  -0.5, 0.5))
  ggml_set_f32(p$r,  runif(ggml_nelements(p$r),   0.0, 1.0))
  ggml_set_f32(p$tf, runif(ggml_nelements(p$tf), -0.5, 0.5))
  # td is a log-decay: negative keeps exp(td) inside the unit interval.
  ggml_set_f32(p$td, runif(ggml_nelements(p$td), -1.0, -0.1))

  out <- ggml_rwkv_wkv6(ctx, p$k, p$v, p$r, p$tf, p$td, p$state)
  y   <- ggml_rwkv_output(ctx, out, p$k)
  compute_1(ctx, out)

  vals <- ggml_get_f32(y)
  expect_length(vals, p$S * p$H * p$n_tokens)
  expect_true(all(is.finite(vals)))
  # With non-zero k, v and r the outputs cannot all be zero.
  expect_gt(max(abs(vals)), 0)
})


# ---------------------------------------------------------------------------
# ssm_conv backward (ggmlR extension -- upstream ggml has no backward for it)
# ---------------------------------------------------------------------------

test_that("ggml_ssm_conv_back packs both input gradients", {
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  d_conv <- 3L; d_inner <- 2L; n_t <- 4L; n_s <- 1L
  len <- d_conv - 1L + n_t
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, len, d_inner, n_s)
  cc <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
  gg <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s)
  for (t in list(sx, cc, gg)) ggml_set_f32(t, rep(0, ggml_nelements(t)))

  bk <- ggml_ssm_conv_back(ctx, sx, cc, gg)
  expect_equal(ggml_nelements(bk),
               ggml_nelements(sx) + ggml_nelements(cc))
})

test_that("ssm_conv gradients match a numeric gradient", {
  # The check the whole backward exists for: an analytic gradient that is
  # subtly wrong still lets a loss descend, so it is compared element by
  # element against a central difference instead.
  set.seed(11L)
  d_conv <- 3L; d_inner <- 2L; n_t <- 4L; n_s <- 2L
  len <- d_conv - 1L + n_t

  sx_v <- runif(len * d_inner * n_s, -1, 1)
  c_v  <- runif(d_conv * d_inner, -1, 1)
  g_v  <- runif(d_inner * n_t * n_s, -1, 1)

  # Forward, computed through ggml, reduced to a scalar by <output, g_v>.
  fwd <- function(sxv, cv) {
    ctx <- ggml_init(32 * 1024 * 1024)
    on.exit(ggml_free(ctx))
    sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, len, d_inner, n_s)
    cc <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
    ggml_set_f32(sx, sxv); ggml_set_f32(cc, cv)
    out <- ggml_ssm_conv(ctx, sx, cc)
    ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, out))
    sum(ggml_get_f32(out) * g_v)
  }

  # Analytic gradients, from the backward op.
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx))
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, len, d_inner, n_s)
  cc <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
  gg <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s)
  ggml_set_f32(sx, sx_v); ggml_set_f32(cc, c_v); ggml_set_f32(gg, g_v)
  bk <- ggml_ssm_conv_back(ctx, sx, cc, gg)
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, bk))
  packed <- ggml_get_f32(bk)
  d_sx <- packed[seq_len(length(sx_v))]
  d_c  <- packed[length(sx_v) + seq_len(length(c_v))]

  # Central difference, one element at a time.
  num_grad <- function(f, p, eps = 1e-3) {
    vapply(seq_along(p), function(i) {
      hi <- p; hi[i] <- hi[i] + eps
      lo <- p; lo[i] <- lo[i] - eps
      (f(hi) - f(lo)) / (2 * eps)
    }, numeric(1))
  }
  n_sx <- num_grad(function(v) fwd(v, c_v), sx_v)
  n_c  <- num_grad(function(v) fwd(sx_v, v), c_v)

  expect_equal(d_sx, n_sx, tolerance = 1e-3)
  expect_equal(d_c,  n_c,  tolerance = 1e-3)
})

test_that("an ssm_conv node differentiates inside a backward graph", {
  # A kernel alone does not make an op trainable -- it also has to be wired
  # into ggml_build_backward_expand(). Upstream's ggml_flash_attn_back() is
  # exactly that trap: the kernel exists, the graph case does not, so nothing
  # differentiates through it.
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  d_conv <- 3L; d_inner <- 2L; n_t <- 4L; n_s <- 1L
  len <- d_conv - 1L + n_t
  sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, len, d_inner, n_s)
  cc <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
  set.seed(5L)
  sx_v <- runif(ggml_nelements(sx), -1, 1)
  c_v  <- runif(ggml_nelements(cc), -1, 1)
  ggml_set_f32(sx, sx_v)
  ggml_set_f32(cc, c_v)
  ggml_set_param(cc)

  loss  <- ggml_sum(ctx, ggml_ssm_conv(ctx, sx, cc))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)   # seeds d(loss)/d(loss) = 1
  ggml_graph_compute(ctx, graph)

  g <- ggml_graph_get_grad(graph, cc)
  expect_false(is.null(g))

  # With loss = sum(output), every output element has upstream gradient 1, so
  # d_c[i0, i1] is just the sum of the sx values that tap it.
  gv  <- ggml_get_f32(g)
  ref <- numeric(d_conv * d_inner)
  for (i1 in seq_len(d_inner)) {
    for (i0 in seq_len(d_conv)) {
      ref[(i1 - 1L) * d_conv + i0] <-
        sum(sx_v[(i1 - 1L) * len + (seq_len(n_t) + i0 - 1L)])
    }
  }
  expect_equal(gv, ref, tolerance = 1e-4)
})

# ---------------------------------------------------------------------------
# ssm_scan backward (ggmlR extension -- upstream ggml has no backward for it)
# ---------------------------------------------------------------------------

test_that("ggml_ssm_scan_back packs all six input gradients", {
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  p <- make_scan(ctx)
  r <- ggml_ssm_scan(ctx, p$s, p$x, p$dt, p$A, p$B, p$C, p$ids)
  g <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(r))
  ggml_set_f32(g, rep(0, ggml_nelements(g)))

  bk <- ggml_ssm_scan_back(ctx, p$s, p$x, p$dt, p$A, p$B, p$C, p$ids, g)

  expect_equal(ggml_nelements(bk),
               p$d_state * p$head_dim * p$n_head * p$n_seqs +   # d_s
               ggml_nelements(p$x)  + ggml_nelements(p$dt) +
               ggml_nelements(p$A)  + ggml_nelements(p$B)  +
               ggml_nelements(p$C))
})

test_that("ssm_scan gradients match a numeric gradient", {
  # The check this whole backward exists for. A BPTT kernel that is subtly
  # wrong still lets a loss descend, so every input is compared element by
  # element against a central difference.
  skip_on_cran()
  set.seed(21L)
  d_state <- 3L; head_dim <- 2L; n_head <- 2L; nt <- 3L; ns <- 1L; ng <- 1L

  vals <- list(
    s  = runif(d_state*head_dim*n_head*ns, -1, 1),
    x  = runif(head_dim*n_head*nt*ns, -1, 1),
    dt = runif(n_head*nt*ns, -1, 1),
    # A must be negative for a decaying (stable) recurrence, as Mamba requires.
    A  = -runif(n_head, 0.2, 1),
    B  = runif(d_state*ng*nt*ns, -1, 1),
    C  = runif(d_state*ng*nt*ns, -1, 1))

  build <- function(ctx, v) {
    s   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, ns)
    x   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, nt, ns)
    dt  <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, nt, ns)
    A   <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_head)   # Mamba-2 form
    B   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, ng, nt, ns)
    Cc  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, ng, nt, ns)
    ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ns)
    ggml_set_f32(s, v$s);  ggml_set_f32(x, v$x);  ggml_set_f32(dt, v$dt)
    ggml_set_f32(A, v$A);  ggml_set_f32(B, v$B);  ggml_set_f32(Cc, v$C)
    ggml_set_i32(ids, seq_len(ns) - 1L)
    list(s = s, x = x, dt = dt, A = A, B = B, C = Cc, ids = ids)
  }

  # Scalar loss = <packed forward result, w>, for a fixed random w. Using the
  # whole packed result means the final state is scored too, not just y.
  n_res <- head_dim*n_head*nt*ns + d_state*head_dim*n_head*ns
  w <- runif(n_res, -1, 1)

  fwd <- function(v) {
    ctx <- ggml_init(64 * 1024 * 1024)
    on.exit(ggml_free(ctx))
    t <- build(ctx, v)
    r <- ggml_ssm_scan(ctx, t$s, t$x, t$dt, t$A, t$B, t$C, t$ids)
    ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, r))
    sum(ggml_get_f32(r) * w)
  }

  # Analytic gradients from the backward op.
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))
  t <- build(ctx, vals)
  r <- ggml_ssm_scan(ctx, t$s, t$x, t$dt, t$A, t$B, t$C, t$ids)
  gt <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_res)
  ggml_set_f32(gt, w)
  bk <- ggml_ssm_scan_back(ctx, t$s, t$x, t$dt, t$A, t$B, t$C, t$ids, gt)
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, bk))
  packed <- ggml_get_f32(bk)

  # Split the packed result in the documented order.
  lens <- vapply(vals, length, integer(1))
  offs <- cumsum(c(0, lens))
  analytic <- lapply(seq_along(vals), function(i)
    packed[(offs[i] + 1L):offs[i + 1L]])
  names(analytic) <- names(vals)

  eps <- 1e-3
  for (nm in names(vals)) {
    num <- vapply(seq_along(vals[[nm]]), function(i) {
      hi <- vals; hi[[nm]][i] <- hi[[nm]][i] + eps
      lo <- vals; lo[[nm]][i] <- lo[[nm]][i] - eps
      (fwd(hi) - fwd(lo)) / (2 * eps)
    }, numeric(1))
    expect_equal(analytic[[nm]], num, tolerance = 2e-2, info = nm)
  }
})

test_that("an ssm_scan node differentiates inside a backward graph", {
  # The graph case, not just the kernel -- see the flash_attn_back trap.
  skip_on_cran()
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  set.seed(23L)
  p <- make_scan(ctx)
  # Non-degenerate operands, and A negative so the recurrence decays.
  ggml_set_f32(p$x,  runif(ggml_nelements(p$x),  -1, 1))
  ggml_set_f32(p$dt, runif(ggml_nelements(p$dt), -1, 1))
  ggml_set_f32(p$A,  -runif(ggml_nelements(p$A), 0.2, 1))
  ggml_set_f32(p$B,  runif(ggml_nelements(p$B),  -1, 1))
  ggml_set_f32(p$C,  runif(ggml_nelements(p$C),  -1, 1))
  ggml_set_param(p$B)

  loss <- ggml_sum(ctx, ggml_ssm_scan(ctx, p$s, p$x, p$dt, p$A, p$B, p$C, p$ids))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)
  ggml_graph_compute(ctx, graph)

  g <- ggml_graph_get_grad(graph, p$B)
  expect_false(is.null(g))
  gv <- ggml_get_f32(g)
  expect_true(all(is.finite(gv)))
  expect_gt(max(abs(gv)), 0)
})

# ---------------------------------------------------------------------------
# Mixed-backend gradients
#
# ggmlR is GPU-first: a training graph runs on Vulkan, and the scheduler moves
# individual ops to the CPU when the backend does not support them. The SSM
# backward kernels have no Vulkan shader (like OUT_PROD and
# CROSS_ENTROPY_LOSS_BACK before them), so they are exactly such ops -- the
# forward runs on the GPU and the backward node falls back.
#
# That split is where mixed-backend bugs live: the scheduler copies split inputs
# INTO a backend but never back out (see the ggmlR notes on sched inplace src
# writes), so an op that reads a GPU-resident tensor from the CPU can silently
# read stale data. These check the gradient is the same either way.
# ---------------------------------------------------------------------------

skip_no_vulkan_ssm <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

test_that("ssm_conv gradients agree between CPU and a scheduled Vulkan graph", {
  skip_on_cran()
  skip_no_vulkan_ssm()

  d_conv <- 3L; d_inner <- 2L; n_t <- 4L; n_s <- 1L
  len <- d_conv - 1L + n_t
  set.seed(31L)
  sx_v <- runif(len * d_inner * n_s, -1, 1)
  c_v  <- runif(d_conv * d_inner, -1, 1)

  run <- function(use_gpu) {
    ctx <- ggml_init(64 * 1024 * 1024)
    on.exit(ggml_free(ctx), add = TRUE)
    ggml_set_no_alloc(ctx, TRUE)

    sx <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, len, d_inner, n_s)
    cc <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner)
    ggml_set_input(sx)
    ggml_set_input(cc)
    ggml_set_param(cc)
    loss <- ggml_sum(ctx, ggml_ssm_conv(ctx, sx, cc))
    ggml_set_loss(loss)

    graph <- ggml_build_forward_expand_grads(ctx, loss)
    ggml_build_backward_expand(ctx, graph)

    # A scheduler, not ggml_backend_graph_compute(): the latter hands the whole
    # graph to one backend, so a node the backend cannot run either aborts or
    # silently produces nothing. The scheduler is what GPU-first training
    # actually uses -- it keeps supported ops on the GPU and moves the rest
    # (OUT_PROD, CROSS_ENTROPY_LOSS_BACK, and these SSM backward nodes) to the
    # CPU, copying tensors across as needed.
    backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
    on.exit(ggml_backend_free(backend), add = TRUE)
    # The scheduler appends a CPU backend itself (ggml requires the last one to
    # be CPU), so passing only the GPU here already gives GPU-first with a CPU
    # fallback -- adding one explicitly would create a second CPU backend.
    sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
    on.exit(ggml_backend_sched_free(sched), add = TRUE)

    ggml_backend_sched_reset(sched)
    ggml_backend_sched_alloc_graph(sched, graph)

    ggml_backend_tensor_set_data(sx, sx_v)
    ggml_backend_tensor_set_data(cc, c_v)
    ggml_graph_reset(graph)
    ggml_backend_sched_graph_compute(sched, graph)

    ggml_backend_tensor_get_data(ggml_graph_get_grad(graph, cc))
  }

  cpu <- run(FALSE)
  gpu <- run(TRUE)
  expect_equal(gpu, cpu, tolerance = 1e-4)
})

test_that("ssm_scan gradients agree between CPU and a scheduled Vulkan graph", {
  skip_on_cran()
  skip_no_vulkan_ssm()

  d_state <- 3L; head_dim <- 2L; n_head <- 2L; nt <- 3L; ns <- 1L; ng <- 1L
  set.seed(33L)
  v <- list(
    s  = runif(d_state*head_dim*n_head*ns, -1, 1),
    x  = runif(head_dim*n_head*nt*ns, -1, 1),
    dt = runif(n_head*nt*ns, -1, 1),
    A  = -runif(n_head, 0.2, 1),
    B  = runif(d_state*ng*nt*ns, -1, 1),
    C  = runif(d_state*ng*nt*ns, -1, 1))

  run <- function(use_gpu) {
    ctx <- ggml_init(128 * 1024 * 1024)
    on.exit(ggml_free(ctx), add = TRUE)
    ggml_set_no_alloc(ctx, TRUE)

    s   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, ns)
    x   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, nt, ns)
    dt  <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, nt, ns)
    A   <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, n_head)
    B   <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, ng, nt, ns)
    Cc  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, ng, nt, ns)
    ids <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ns)
    for (t in list(s, x, dt, A, B, Cc, ids)) ggml_set_input(t)
    ggml_set_param(B)

    loss <- ggml_sum(ctx, ggml_ssm_scan(ctx, s, x, dt, A, B, Cc, ids))
    ggml_set_loss(loss)
    graph <- ggml_build_forward_expand_grads(ctx, loss)
    ggml_build_backward_expand(ctx, graph)

    backend <- if (use_gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
    on.exit(ggml_backend_free(backend), add = TRUE)
    sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
    on.exit(ggml_backend_sched_free(sched), add = TRUE)

    ggml_backend_sched_reset(sched)
    ggml_backend_sched_alloc_graph(sched, graph)

    ggml_backend_tensor_set_data(s,  v$s);  ggml_backend_tensor_set_data(x, v$x)
    ggml_backend_tensor_set_data(dt, v$dt); ggml_backend_tensor_set_data(A, v$A)
    ggml_backend_tensor_set_data(B,  v$B);  ggml_backend_tensor_set_data(Cc, v$C)
    ggml_backend_tensor_set_data(ids, as.integer(seq_len(ns) - 1L))

    ggml_graph_reset(graph)
    ggml_backend_sched_graph_compute(sched, graph)
    ggml_backend_tensor_get_data(ggml_graph_get_grad(graph, B))
  }

  cpu <- run(FALSE)
  gpu <- run(TRUE)
  expect_equal(gpu, cpu, tolerance = 1e-3)
})

# ---------------------------------------------------------------------------
# rwkv_wkv6 backward (ggmlR extension -- upstream has no backward for it)
# ---------------------------------------------------------------------------

test_that("rwkv_wkv6 gradients match a numeric gradient", {
  skip_on_cran()
  set.seed(41L)
  S <- 3L; H <- 1L; T <- 4L; ns <- 1L

  v0 <- list(
    k  = runif(S*H*T, -0.5, 0.5),
    v  = runif(S*H*T, -0.5, 0.5),
    r  = runif(S*H*T,  0.0, 1.0),
    tf = runif(S*H,   -0.5, 0.5),
    # td is a decay factor in (0,1): the forward multiplies the state by it.
    td = exp(runif(S*H*T, -1.0, -0.1)),
    s  = runif(S*S*H*ns, -0.5, 0.5))

  build <- function(ctx, v) {
    mk3 <- function(x) {
      t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, T); ggml_set_f32(t, x); t
    }
    tf <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, H); ggml_set_f32(tf, v$tf)
    st <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*S*H, ns); ggml_set_f32(st, v$s)
    list(k = mk3(v$k), v = mk3(v$v), r = mk3(v$r), tf = tf, td = mk3(v$td), s = st)
  }

  n_res <- S*H*(T + S*ns)
  w <- runif(n_res, -1, 1)

  fwd <- function(v) {
    ctx <- ggml_init(64 * 1024 * 1024); on.exit(ggml_free(ctx))
    t <- build(ctx, v)
    out <- ggml_rwkv_wkv6(ctx, t$k, t$v, t$r, t$tf, t$td, t$s)
    ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, out))
    sum(ggml_get_f32(out) * w)
  }

  ctx <- ggml_init(64 * 1024 * 1024); on.exit(ggml_free(ctx))
  t <- build(ctx, v0)
  gt <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_res); ggml_set_f32(gt, w)
  bk <- ggml_rwkv_wkv6_back(ctx, t$k, t$v, t$r, t$tf, t$td, t$s, gt)
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, bk))
  packed <- ggml_get_f32(bk)

  lens <- vapply(v0, length, integer(1))
  offs <- cumsum(c(0, lens))
  analytic <- lapply(seq_along(v0), function(i) packed[(offs[i]+1L):offs[i+1L]])
  names(analytic) <- names(v0)

  eps <- 1e-3
  for (nm in names(v0)) {
    num <- vapply(seq_along(v0[[nm]]), function(i) {
      hi <- v0; hi[[nm]][i] <- hi[[nm]][i] + eps
      lo <- v0; lo[[nm]][i] <- lo[[nm]][i] - eps
      (fwd(hi) - fwd(lo)) / (2 * eps)
    }, numeric(1))
    expect_equal(analytic[[nm]], num, tolerance = 2e-2, info = nm)
  }
})

test_that("an rwkv_wkv6 node differentiates inside a backward graph", {
  skip_on_cran()
  ctx <- ggml_init(64 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  set.seed(43L)
  p <- make_rwkv(ctx)
  ggml_set_f32(p$k,  runif(ggml_nelements(p$k),  -0.5, 0.5))
  ggml_set_f32(p$v,  runif(ggml_nelements(p$v),  -0.5, 0.5))
  ggml_set_f32(p$r,  runif(ggml_nelements(p$r),   0.0, 1.0))
  ggml_set_f32(p$tf, runif(ggml_nelements(p$tf), -0.5, 0.5))
  ggml_set_f32(p$td, exp(runif(ggml_nelements(p$td), -1.0, -0.1)))
  ggml_set_param(p$k)

  loss <- ggml_sum(ctx, ggml_rwkv_wkv6(ctx, p$k, p$v, p$r, p$tf, p$td, p$state))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)
  ggml_graph_compute(ctx, graph)

  g <- ggml_graph_get_grad(graph, p$k)
  expect_false(is.null(g))
  gv <- ggml_get_f32(g)
  expect_true(all(is.finite(gv)))
  expect_gt(max(abs(gv)), 0)
})

# ---------------------------------------------------------------------------
# rwkv_wkv7 and gated_linear_attn backward (ggmlR extensions)
# ---------------------------------------------------------------------------

# Shared driver: build the forward op from a named list of vectors, contract the
# packed result with a fixed random w to get a scalar, and compare the packed
# analytic gradient against a central difference for every input.
check_recurrent_grad <- function(names_order, build_fwd, build_back, v0,
                                 n_res, tol = 3e-2) {
  # n_res arrives as a promise evaluated HERE, not at the call site, so any
  # variable it names must resolve in this frame or above -- a caller using a
  # local `T` gets base::T (TRUE, i.e. 1) instead, silently undersizing the
  # gradient tensor and making the backward op read past it. Force it early and
  # check it against the forward result rather than trusting the arithmetic.
  n_res <- as.integer(n_res)
  w <- runif(n_res, -1, 1)

  fwd <- function(v) {
    ctx <- ggml_init(64 * 1024 * 1024); on.exit(ggml_free(ctx))
    out <- build_fwd(ctx, v)
    ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, out))
    sum(ggml_get_f32(out) * w)
  }

  ctx <- ggml_init(64 * 1024 * 1024); on.exit(ggml_free(ctx))

  # Guard the size before it can matter: the backward op reads exactly as many
  # elements as the forward produced, so a wrong n_res is an out-of-bounds read
  # that surfaces as NaN or a crash rather than as a failed expectation.
  probe <- build_fwd(ctx, v0)
  if (ggml_nelements(probe) != n_res) {
    stop(sprintf("n_res is %d but the forward op produces %d elements",
                 n_res, ggml_nelements(probe)))
  }

  gt <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_res); ggml_set_f32(gt, w)
  bk <- build_back(ctx, v0, gt)
  ggml_graph_compute(ctx, ggml_build_forward_expand(ctx, bk))
  packed <- ggml_get_f32(bk)

  lens <- vapply(v0[names_order], length, integer(1))
  offs <- cumsum(c(0, lens))

  eps <- 1e-3
  for (i in seq_along(names_order)) {
    nm <- names_order[i]
    analytic <- packed[(offs[i] + 1L):offs[i + 1L]]
    num <- vapply(seq_along(v0[[nm]]), function(j) {
      hi <- v0; hi[[nm]][j] <- hi[[nm]][j] + eps
      lo <- v0; lo[[nm]][j] <- lo[[nm]][j] - eps
      (fwd(hi) - fwd(lo)) / (2 * eps)
    }, numeric(1))
    expect_equal(analytic, num, tolerance = tol, info = nm)
  }
}

test_that("rwkv_wkv7 gradients match a numeric gradient", {
  skip_on_cran()
  set.seed(51L)
  S <- 3L; H <- 1L; n_tok <- 3L; ns <- 1L
  # n_tok, not T: `n_res` below is a lazy argument evaluated inside
  # check_recurrent_grad(), where a local `T` is not in scope and R falls back
  # to base::T == TRUE == 1. That silently sized the gradient tensor 12 instead
  # of 18, so the backward op read past it -- NaN, and a segfault before that.
  mk <- function() runif(S*H*n_tok, -0.4, 0.4)

  v0 <- list(r = mk(), w = exp(runif(S*H*n_tok, -1, -0.1)), k = mk(), v = mk(),
             a = runif(S*H*n_tok, -0.3, 0.3), b = runif(S*H*n_tok, -0.3, 0.3),
             s = runif(S*S*H*ns, -0.4, 0.4))

  t3 <- function(ctx, x) {
    t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tok); ggml_set_f32(t, x); t
  }
  mk_all <- function(ctx, v) {
    st <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*S*H, ns); ggml_set_f32(st, v$s)
    list(r = t3(ctx, v$r), w = t3(ctx, v$w), k = t3(ctx, v$k), v = t3(ctx, v$v),
         a = t3(ctx, v$a), b = t3(ctx, v$b), s = st)
  }

  check_recurrent_grad(
    c("r", "w", "k", "v", "a", "b", "s"),
    function(ctx, v) { t <- mk_all(ctx, v)
      ggml_rwkv_wkv7(ctx, t$r, t$w, t$k, t$v, t$a, t$b, t$s) },
    function(ctx, v, gt) { t <- mk_all(ctx, v)
      ggml_rwkv_wkv7_back(ctx, t$r, t$w, t$k, t$v, t$a, t$b, t$s, gt) },
    v0, n_res = S*H*(n_tok + S*ns))
})

test_that("gated_linear_attn gradients match a numeric gradient", {
  skip_on_cran()
  set.seed(53L)
  S <- 3L; H <- 1L; n_tok <- 3L; ns <- 1L; sc <- 0.7
  mk <- function() runif(S*H*n_tok, -0.4, 0.4)

  v0 <- list(k = mk(), v = mk(), q = mk(),
             g = exp(runif(S*H*n_tok, -1, -0.1)), s = runif(S*S*H*ns, -0.4, 0.4))

  t3 <- function(ctx, x) {
    t <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tok); ggml_set_f32(t, x); t
  }
  mk_all <- function(ctx, v) {
    st <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S*S*H, ns); ggml_set_f32(st, v$s)
    list(k = t3(ctx, v$k), v = t3(ctx, v$v), q = t3(ctx, v$q),
         g = t3(ctx, v$g), s = st)
  }

  check_recurrent_grad(
    c("k", "v", "q", "g", "s"),
    function(ctx, v) { t <- mk_all(ctx, v)
      ggml_gated_linear_attn(ctx, t$k, t$v, t$q, t$g, t$s, sc) },
    function(ctx, v, gt) { t <- mk_all(ctx, v)
      ggml_gated_linear_attn_back(ctx, t$k, t$v, t$q, t$g, t$s, gt, sc) },
    v0, n_res = S*H*(n_tok + S*ns))
})

test_that("wkv7 and GLA nodes differentiate inside a backward graph", {
  skip_on_cran()
  for (which_op in c("wkv7", "gla")) {
    ctx <- ggml_init(64 * 1024 * 1024)
    set.seed(57L)
    p <- make_rwkv(ctx)
    for (t in list(p$k, p$v, p$r, p$w, p$a, p$b, p$q)) {
      ggml_set_f32(t, runif(ggml_nelements(t), -0.4, 0.4))
    }
    ggml_set_f32(p$g, exp(runif(ggml_nelements(p$g), -1, -0.1)))
    ggml_set_param(p$k)

    out <- if (which_op == "wkv7") {
      ggml_rwkv_wkv7(ctx, p$r, p$w, p$k, p$v, p$a, p$b, p$state)
    } else {
      ggml_gated_linear_attn(ctx, p$k, p$v, p$q, p$g, p$state, 1 / sqrt(p$S))
    }
    loss <- ggml_sum(ctx, out)
    ggml_set_loss(loss)
    graph <- ggml_build_forward_expand_grads(ctx, loss)
    ggml_build_backward_expand(ctx, graph)
    ggml_graph_reset(graph)
    ggml_graph_compute(ctx, graph)

    g <- ggml_graph_get_grad(graph, p$k)
    expect_false(is.null(g), info = which_op)
    expect_true(all(is.finite(ggml_get_f32(g))), info = which_op)
    ggml_free(ctx)
  }
})
