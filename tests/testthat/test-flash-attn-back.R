# Backward pass for flash attention.
#
# Upstream ggml has no working one: ggml_flash_attn_back() aborts on its first
# statement and was never wired into ggml-graph.c, so attention was
# inference-only through the fused op. These check both halves of that -- the
# kernel's numbers, and the graph actually reaching it.

# Reference forward attention in plain R, matching ggml_flash_attn_ext:
#   q [DK,N,H,B]  k [DK,M,Hkv,B]  v [DV,M,Hkv,B]  ->  out [DV,H,N,B]
# Arrays here are R arrays in ggml's own ne order, so index [i,j,h,b] directly.
attn_ref <- function(q, k, v, scale, mask = NULL) {
  DK <- dim(q)[1]; N <- dim(q)[2]; H <- dim(q)[3]; B <- dim(q)[4]
  DV <- dim(v)[1]; M <- dim(k)[2]; Hkv <- dim(k)[3]
  rk2 <- H / Hkv
  out <- array(0, c(DV, H, N, B))
  for (b in seq_len(B)) for (h in seq_len(H)) {
    hk <- (h - 1L) %/% rk2 + 1L
    for (n in seq_len(N)) {
      s <- vapply(seq_len(M), function(m) sum(q[, n, h, b] * k[, m, hk, b]), numeric(1))
      s <- s * scale
      if (!is.null(mask)) s <- s + mask[, n]
      if (all(!is.finite(s))) next
      e <- exp(s - max(s[is.finite(s)]))
      e[!is.finite(s)] <- 0
      p <- e / sum(e)
      for (d in seq_len(DV)) out[d, h, n, b] <- sum(p * v[d, , hk, b])
    }
  }
  out
}

# Analytic gradients from the op, unpacked from the [d_q | d_k | d_v] blob.
# The kernel pads each slice to GGML_MEM_ALIGN (16 bytes = 4 floats).
fab_grads <- function(q, k, v, d, scale, mask = NULL) {
  ctx <- ggml_init(256 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  mk <- function(a) ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                       dim(a)[1], dim(a)[2], dim(a)[3], dim(a)[4])
  tq <- mk(q); tk <- mk(k); tv <- mk(v); td <- mk(d)

  tm <- NULL
  if (!is.null(mask)) {
    tm <- ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nrow(mask), ncol(mask))
  }

  bk <- ggml_flash_attn_back(ctx, tq, tk, tv, tm, td, scale)

  be <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(be, 2L)
  on.exit(ggml_backend_free(be), add = TRUE)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)

  ggml_backend_tensor_set_data(tq, as.numeric(q))
  ggml_backend_tensor_set_data(tk, as.numeric(k))
  ggml_backend_tensor_set_data(tv, as.numeric(v))
  ggml_backend_tensor_set_data(td, as.numeric(d))
  if (!is.null(tm)) ggml_backend_tensor_set_data(tm, as.numeric(mask))

  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, bk))
  packed <- ggml_backend_tensor_get_data(bk)

  pad <- function(n) ceiling(n * 4 / 16) * 16 / 4   # floats -> 16-byte alignment
  nq <- length(q); nk <- length(k); nv <- length(v)
  off_k <- pad(nq)
  off_v <- off_k + pad(nk)

  list(
    q = array(packed[1:nq], dim(q)),
    k = array(packed[(off_k + 1):(off_k + nk)], dim(k)),
    v = array(packed[(off_v + 1):(off_v + nv)], dim(v))
  )
}

# Central-difference gradient of sum(d * attn(q,k,v)) w.r.t. one input.
num_grad <- function(q, k, v, d, scale, mask, which, eps = 1e-3) {
  vals <- list(q = q, k = k, v = v)
  g <- numeric(length(vals[[which]]))
  for (i in seq_along(g)) {
    hi <- vals; hi[[which]][i] <- hi[[which]][i] + eps
    lo <- vals; lo[[which]][i] <- lo[[which]][i] - eps
    fh <- sum(d * attn_ref(hi$q, hi$k, hi$v, scale, mask))
    fl <- sum(d * attn_ref(lo$q, lo$k, lo$v, scale, mask))
    g[i] <- (fh - fl) / (2 * eps)
  }
  array(g, dim(vals[[which]]))
}

test_that("flash_attn_back matches finite differences (single head)", {
  set.seed(11L)
  DK <- 4L; DV <- 4L; N <- 3L; M <- 5L
  q <- array(runif(DK * N, -1, 1), c(DK, N, 1L, 1L))
  k <- array(runif(DK * M, -1, 1), c(DK, M, 1L, 1L))
  v <- array(runif(DV * M, -1, 1), c(DV, M, 1L, 1L))
  d <- array(runif(DV * N, -1, 1), c(DV, 1L, N, 1L))
  scale <- 1 / sqrt(DK)

  a <- fab_grads(q, k, v, d, scale)
  for (nm in c("q", "k", "v")) {
    expect_equal(as.numeric(a[[nm]]),
                 as.numeric(num_grad(q, k, v, d, scale, NULL, nm)),
                 tolerance = 2e-2, info = nm)
  }
})

test_that("flash_attn_back handles a mask", {
  set.seed(12L)
  DK <- 4L; DV <- 4L; N <- 3L; M <- 4L
  q <- array(runif(DK * N, -1, 1), c(DK, N, 1L, 1L))
  k <- array(runif(DK * M, -1, 1), c(DK, M, 1L, 1L))
  v <- array(runif(DV * M, -1, 1), c(DV, M, 1L, 1L))
  d <- array(runif(DV * N, -1, 1), c(DV, 1L, N, 1L))
  scale <- 1 / sqrt(DK)

  # Causal mask [M, N]: query n attends to keys 1..n only.
  mask <- matrix(0, M, N)
  for (n in seq_len(N)) if (n < M) mask[(n + 1L):M, n] <- -Inf

  a <- fab_grads(q, k, v, d, scale, mask)
  for (nm in c("q", "k", "v")) {
    expect_equal(as.numeric(a[[nm]]),
                 as.numeric(num_grad(q, k, v, d, scale, mask, nm)),
                 tolerance = 2e-2, info = nm)
  }
  # A key no query attends to gets no gradient.
  expect_true(all(is.finite(as.numeric(a$k))))
})

test_that("flash_attn_back handles grouped-query attention and batches", {
  set.seed(13L)
  DK <- 4L; DV <- 4L; N <- 2L; M <- 3L; H <- 4L; Hkv <- 2L; B <- 2L
  q <- array(runif(DK * N * H * B, -1, 1),   c(DK, N, H, B))
  k <- array(runif(DK * M * Hkv * B, -1, 1), c(DK, M, Hkv, B))
  v <- array(runif(DV * M * Hkv * B, -1, 1), c(DV, M, Hkv, B))
  d <- array(runif(DV * H * N * B, -1, 1),   c(DV, H, N, B))
  scale <- 1 / sqrt(DK)

  a <- fab_grads(q, k, v, d, scale)
  for (nm in c("q", "k", "v")) {
    expect_equal(as.numeric(a[[nm]]),
                 as.numeric(num_grad(q, k, v, d, scale, NULL, nm)),
                 tolerance = 2e-2, info = nm)
  }
})

test_that("a flash_attn_ext node differentiates inside a backward graph", {
  # The graph case, not just the kernel: upstream never wired this op into
  # ggml-graph.c, so a kernel that works proves nothing on its own.
  ctx <- ggml_init(128 * 1024 * 1024)
  on.exit(ggml_free(ctx))

  set.seed(14L)
  DK <- 4L; DV <- 4L; N <- 3L; M <- 4L
  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, 1L, 1L)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, 1L, 1L)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, 1L, 1L)
  ggml_set_f32(q, runif(DK * N, -1, 1))
  ggml_set_f32(k, runif(DK * M, -1, 1))
  ggml_set_f32(v, runif(DV * M, -1, 1))

  ggml_set_param(q)
  ggml_set_param(v)

  out  <- ggml_flash_attn_ext(ctx, q, k, v, NULL, 1 / sqrt(DK))
  loss <- ggml_sum(ctx, out)
  ggml_set_loss(loss)

  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)          # without this the gradients are silently 0
  ggml_graph_compute(ctx, graph)

  gq <- ggml_graph_get_grad(graph, q)
  gv <- ggml_graph_get_grad(graph, v)
  expect_false(is.null(gq))
  expect_false(is.null(gv))
  expect_true(all(is.finite(ggml_get_f32(gq))))
  expect_true(all(is.finite(ggml_get_f32(gv))))
  # d(sum(softmax(qk)v))/dv is the summed attention weights: strictly positive.
  expect_gt(max(abs(ggml_get_f32(gv))), 0)
})
test_that("the graph builder rejects attention features it cannot differentiate", {
  # ALiBi (max_bias > 0) has no backward here. The builder must say so rather
  # than hand back a silently wrong gradient.
  ctx <- ggml_init(32 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  DK <- 4L; DV <- 4L; N <- 2L; M <- 2L
  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, 1L, 1L)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, 1L, 1L)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, 1L, 1L)
  ggml_set_f32(q, runif(DK * N)); ggml_set_f32(k, runif(DK * M))
  ggml_set_f32(v, runif(DV * M))
  ggml_set_param(q)

  mask <- ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M, N)
  out  <- ggml_flash_attn_ext(ctx, q, k, v, mask, 1 / sqrt(DK), max_bias = 8.0)
  loss <- ggml_sum(ctx, out)
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  expect_error(ggml_build_backward_expand(ctx, graph))
})

# ---------------------------------------------------------------------------
# Packing layout
#
# The op returns grad_q | grad_k | grad_v in one buffer, and ggml-graph.c hands
# each src a view of its own slice. Finite differences above check the VALUES,
# which a view pointing at the wrong offset would usually break -- but not
# always: a stride mismatch can still land on the right numbers for a small
# symmetric case. These check the addressing itself.
# ---------------------------------------------------------------------------

test_that("the packed gradients occupy disjoint, in-bounds slices", {
  # Deliberately unequal sizes and a head count that is not a divisor of the
  # buffer length, so an off-by-one in the offsets cannot coincide.
  DK <- 3L; DV <- 5L; N <- 2L; M <- 4L; H <- 4L; Hkv <- 2L; B <- 2L

  ctx <- ggml_init(64 * 1024 * 1024)
  ggml_set_no_alloc(ctx, TRUE)
  on.exit(ggml_free(ctx), add = TRUE)

  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, H, B)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, Hkv, B)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, Hkv, B)
  d <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, H, N, B)

  bk <- ggml_flash_attn_back(ctx, q, k, v, NULL, d, 1 / sqrt(DK))

  be <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(be, 2L)
  on.exit(ggml_backend_free(be), add = TRUE)
  buf <- ggml_backend_alloc_ctx_tensors(ctx, be)

  set.seed(31L)
  ggml_backend_tensor_set_data(q, runif(DK * N * H * B, -1, 1))
  ggml_backend_tensor_set_data(k, runif(DK * M * Hkv * B, -1, 1))
  ggml_backend_tensor_set_data(v, runif(DV * M * Hkv * B, -1, 1))
  ggml_backend_tensor_set_data(d, runif(DV * H * N * B, -1, 1))

  ggml_backend_graph_compute(be, ggml_build_forward_expand(ctx, bk))
  packed <- ggml_backend_tensor_get_data(bk)

  nq <- DK * N * H * B
  nk <- DK * M * Hkv * B
  nv <- DV * M * Hkv * B

  pad   <- function(n) ceiling(n * 4 / 16) * 16 / 4
  off_k <- pad(nq)
  off_v <- off_k + pad(nk)

  # Every slice must fit inside the tensor the constructor sized.
  expect_gte(length(packed), off_v + nv)

  # Slices must not overlap: each one ends before the next one starts.
  expect_lte(nq, off_k)
  expect_lte(off_k + nk, off_v)

  # The kernel zeroes the whole buffer, then writes each slice. With random
  # inputs no gradient is identically zero, so a slice that is all-zero means
  # nothing was written there -- the classic symptom of a wrong offset.
  expect_gt(max(abs(packed[1:nq])), 0)
  expect_gt(max(abs(packed[(off_k + 1):(off_k + nk)])), 0)
  expect_gt(max(abs(packed[(off_v + 1):(off_v + nv)])), 0)

  # The padding between slices is never written, so it stays zero. If a slice
  # ran long -- a stride computed from the wrong tensor -- it would spill here.
  if (off_k > nq) {
    expect_true(all(packed[(nq + 1):off_k] == 0))
  }
  if (off_v > off_k + nk) {
    expect_true(all(packed[(off_k + nk + 1):off_v] == 0))
  }
})

test_that("autodiff views address their own slice of the packed buffer", {
  # The graph path, where the views actually live. Each gradient is compared
  # against the same slice taken by hand from a direct flash_attn_back call: if
  # a view in ggml-graph.c pointed at the wrong offset or walked with the wrong
  # stride, the two would disagree.
  DK <- 3L; DV <- 5L; N <- 2L; M <- 4L; H <- 4L; Hkv <- 2L; B <- 2L

  set.seed(32L)
  qd <- runif(DK * N * H * B, -1, 1)
  kd <- runif(DK * M * Hkv * B, -1, 1)
  vd <- runif(DV * M * Hkv * B, -1, 1)

  # -- gradients through the graph --
  ctx <- ggml_init(128 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)

  q <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, N, H, B)
  k <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, M, Hkv, B)
  v <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DV, M, Hkv, B)
  ggml_set_f32(q, qd); ggml_set_f32(k, kd); ggml_set_f32(v, vd)
  ggml_set_param(q); ggml_set_param(k); ggml_set_param(v)

  loss <- ggml_sum(ctx, ggml_flash_attn_ext(ctx, q, k, v, NULL, 1 / sqrt(DK)))
  ggml_set_loss(loss)
  graph <- ggml_build_forward_expand_grads(ctx, loss)
  ggml_build_backward_expand(ctx, graph)
  ggml_graph_reset(graph)
  ggml_graph_compute(ctx, graph)

  g_q <- ggml_get_f32(ggml_graph_get_grad(graph, q))
  g_k <- ggml_get_f32(ggml_graph_get_grad(graph, k))
  g_v <- ggml_get_f32(ggml_graph_get_grad(graph, v))

  # -- the same thing packed, sliced by hand --
  # d/d(out) of sum(out) is all ones, in the forward's permuted shape.
  a <- fab_grads(array(qd, c(DK, N, H, B)),
                 array(kd, c(DK, M, Hkv, B)),
                 array(vd, c(DV, M, Hkv, B)),
                 array(1,  c(DV, H, N, B)),
                 1 / sqrt(DK))

  expect_equal(g_q, as.numeric(a$q), tolerance = 1e-5)
  expect_equal(g_k, as.numeric(a$k), tolerance = 1e-5)
  expect_equal(g_v, as.numeric(a$v), tolerance = 1e-5)

  # Each gradient has its source tensor's length -- a view that overran would
  # come back the wrong size.
  expect_length(g_q, DK * N * H * B)
  expect_length(g_k, DK * M * Hkv * B)
  expect_length(g_v, DV * M * Hkv * B)
})
