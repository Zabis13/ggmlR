# Tests for ag_flash_attention() -- multi-head attention as one fused op.
#
# The reference here is finite differences, not the existing head loop. That is
# deliberate: while this was being written the head loop itself turned out to
# produce gradients scaled by the number of heads (see the regression tests in
# test-autograd.R), so checking one implementation against the other would have
# confirmed a shared bug rather than catching it. Numerical differentiation
# depends on neither.
#
# The layouts are where this op is easy to get wrong -- flash attention works in
# 3D ([d_head, seq, n_head]) with a PERMUTED result ([d_head, n_head, seq]),
# while ag_* is 2D throughout, and ggml_flash_attn_back returns dq/dk/dv packed
# into one buffer with 16-byte alignment between the slices. A misread offset
# shifts dk and dv without changing dq, so each gradient is checked separately.

ag_flash_attention <- ggmlR:::ag_flash_attention

# Pin a CPU backend for these tests.
#
# These comparisons check layout and correctness, not which device runs them,
# and the references are computed in double precision -- so anything executing
# on Vulkan disagrees at f16 level (~1e-3) for reasons that have nothing to do
# with what is being tested. Pinning the device is the fix; loosening the
# tolerance to 1e-3 instead would also stop the tests catching a genuinely
# wrong permutation, which is exactly what they exist for.
#
# The original cause was ag_device("cpu") not releasing a leaked Vulkan
# backend, which is fixed (see test-ag-device.R). This stays anyway: it makes
# the file independent of whatever any other test leaves in the device state,
# rather than of one particular bug.
with_cpu_backend <- function(code) {
  st <- ggmlR:::.ag_device_state
  prev_dev <- st$device
  prev_bk  <- st$backend
  on.exit({
    st$device  <- prev_dev
    st$backend <- prev_bk
  }, add = TRUE)

  st$device  <- "cpu"
  st$backend <- ggml_backend_cpu_init()
  ggml_backend_cpu_set_n_threads(st$backend, ggml_get_n_threads())

  # Evaluate in the caller's frame: the block references the test's own
  # variables, which a plain force() would look for here instead.
  eval(substitute(code), envir = parent.frame())
}

# Attention in plain R, over the same head layout: rows of a [d_model, seq]
# matrix split into n_heads contiguous blocks.
attn_ref <- function(q, k, v, n_heads) {
  d <- nrow(q); s <- ncol(q); dh <- d %/% n_heads
  scale <- 1 / sqrt(dh)
  out <- matrix(0, d, ncol(q))
  for (i in seq_len(n_heads)) {
    lo <- (i - 1L) * dh + 1L; hi <- i * dh
    sc <- t(q[lo:hi, , drop = FALSE]) %*% k[lo:hi, , drop = FALSE] * scale
    p  <- t(apply(sc, 1, function(r) exp(r - max(r)) / sum(exp(r - max(r)))))
    out[lo:hi, ] <- v[lo:hi, , drop = FALSE] %*% t(p)
  }
  out
}

# Central-difference gradient of sum(G * attn(q,k,v)) w.r.t. one input.
num_grad <- function(Q, K, V, G, n_heads, which, eps = 1e-5) {
  M <- list(q = Q, k = K, v = V)
  g <- numeric(length(M[[which]]))
  for (i in seq_along(g)) {
    hi <- M; hi[[which]][i] <- hi[[which]][i] + eps
    lo <- M; lo[[which]][i] <- lo[[which]][i] - eps
    g[i] <- (sum(G * attn_ref(hi$q, hi$k, hi$v, n_heads)) -
             sum(G * attn_ref(lo$q, lo$k, lo$v, n_heads))) / (2 * eps)
  }
  matrix(g, nrow(M[[which]]), ncol(M[[which]]))
}

test_that("flash attention matches a reference implementation", {
  with_cpu_backend({
  for (cfg in list(c(d = 8, s = 4, h = 1), c(d = 8, s = 4, h = 2),
                   c(d = 16, s = 6, h = 4))) {
    d <- as.integer(cfg[["d"]]); s <- as.integer(cfg[["s"]])
    h <- as.integer(cfg[["h"]])
    set.seed(23L)
    Q <- matrix(runif(d * s, -1, 1), d, s)
    K <- matrix(runif(d * s, -1, 1), d, s)
    V <- matrix(runif(d * s, -1, 1), d, s)

    out <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h)

    expect_equal(dim(ggmlR:::.ag_data(out)), c(d, s))
    expect_equal(ggmlR:::.ag_data(out), attn_ref(Q, K, V, h),
                 tolerance = 1e-5, info = paste("d", d, "h", h))
  }
  })
})

test_that("flash attention gradients match finite differences", {
  with_cpu_backend({
  d <- 8L; s <- 4L; h <- 2L
  set.seed(7L)
  Q <- matrix(runif(d * s, -1, 1), d, s)
  K <- matrix(runif(d * s, -1, 1), d, s)
  V <- matrix(runif(d * s, -1, 1), d, s)
  G <- matrix(runif(d * s, -1, 1), d, s)

  q <- ag_param(Q); k <- ag_param(K); v <- ag_param(V)
  with_grad_tape({
    out  <- ag_flash_attention(q, k, v, h)
    loss <- ag_sum(ag_mul(out, ag_tensor(G)))     # dL/dout = G exactly
  })
  backward(loss)

  # Checked one at a time: the three gradients come back packed in a single
  # buffer, so an offset error moves dk and dv while leaving dq correct.
  expect_equal(q$grad, num_grad(Q, K, V, G, h, "q"), tolerance = 1e-5)
  expect_equal(k$grad, num_grad(Q, K, V, G, h, "k"), tolerance = 1e-5)
  expect_equal(v$grad, num_grad(Q, K, V, G, h, "v"), tolerance = 1e-5)
  })
})

test_that("flash attention collapses the tape to one node", {
  # The point of the op: ag_multihead_attention records a node per slice,
  # score, softmax and matmul per head, flash records one.
  with_cpu_backend({
  d <- 16L; s <- 8L; h <- 4L
  set.seed(41L)
  q <- ag_param(matrix(runif(d * s, -1, 1), d, s))
  k <- ag_param(matrix(runif(d * s, -1, 1), d, s))
  v <- ag_param(matrix(runif(d * s, -1, 1), d, s))

  with_grad_tape({
    invisible(ag_flash_attention(q, k, v, h))
  })
  expect_equal(length(ggmlR:::.ag_tape$nodes), 1L)
  })
})

test_that("flash attention handles cross-attention shapes", {
  # k and v may be a different length from q; they must match each other.
  with_cpu_backend({
  d <- 8L; sq <- 3L; skv <- 5L; h <- 2L
  set.seed(29L)
  Q <- matrix(runif(d * sq, -1, 1), d, sq)
  K <- matrix(runif(d * skv, -1, 1), d, skv)
  V <- matrix(runif(d * skv, -1, 1), d, skv)

  out <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h)
  expect_equal(dim(ggmlR:::.ag_data(out)), c(d, sq))

  # Reference for unequal lengths.
  dh <- d %/% h; scale <- 1 / sqrt(dh)
  want <- matrix(0, d, sq)
  for (i in seq_len(h)) {
    lo <- (i - 1L) * dh + 1L; hi <- i * dh
    sc <- t(Q[lo:hi, , drop = FALSE]) %*% K[lo:hi, , drop = FALSE] * scale
    p  <- t(apply(sc, 1, function(r) exp(r - max(r)) / sum(exp(r - max(r)))))
    want[lo:hi, ] <- V[lo:hi, , drop = FALSE] %*% t(p)
  }
  expect_equal(ggmlR:::.ag_data(out), want, tolerance = 1e-5)
  })
})

test_that("flash attention rejects mismatched shapes", {
  with_cpu_backend({
  q <- ag_tensor(matrix(0, 8, 4))
  k <- ag_tensor(matrix(0, 8, 4))
  v <- ag_tensor(matrix(0, 8, 4))

  expect_error(ag_flash_attention(q, k, v, 3L), "divide d_model")
  expect_error(ag_flash_attention(q, ag_tensor(matrix(0, 6, 4)), v, 2L),
               "share d_model")
  expect_error(ag_flash_attention(q, k, ag_tensor(matrix(0, 8, 6)), 2L),
               "sequence length")
  })
})

# Attention with a mask, in plain R. `blocked(j)` returns the additive mask for
# query j: 0 where attention is allowed, -Inf where it is not.
attn_ref_masked <- function(q, k, v, n_heads, blocked) {
  d <- nrow(q); sq <- ncol(q); skv <- ncol(k); dh <- d %/% n_heads
  scale <- 1 / sqrt(dh)
  out <- matrix(0, d, sq)
  for (i in seq_len(n_heads)) {
    lo <- (i - 1L) * dh + 1L; hi <- i * dh
    sc <- t(q[lo:hi, , drop = FALSE]) %*% k[lo:hi, , drop = FALSE] * scale
    for (j in seq_len(sq)) sc[j, ] <- sc[j, ] + blocked(j)
    p <- t(apply(sc, 1, function(r) {
      f <- is.finite(r); ex <- rep(0, length(r))
      ex[f] <- exp(r[f] - max(r[f])); ex / sum(ex)
    }))
    out[lo:hi, ] <- v[lo:hi, , drop = FALSE] %*% t(p)
  }
  out
}

test_that("causal = TRUE masks future positions", {
  with_cpu_backend({
    d <- 8L; s <- 4L; h <- 2L
    set.seed(31L)
    Q <- matrix(runif(d * s, -1, 1), d, s)
    K <- matrix(runif(d * s, -1, 1), d, s)
    V <- matrix(runif(d * s, -1, 1), d, s)

    got <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h,
                              causal = TRUE)
    want <- attn_ref_masked(Q, K, V, h,
                            function(j) ifelse(seq_len(s) <= j, 0, -Inf))

    expect_equal(ggmlR:::.ag_data(got), want, tolerance = 1e-5)

    # And it must actually differ from the unmasked result -- a mask that is
    # silently dropped would still pass an "equals the reference" test if the
    # reference were computed without one.
    plain <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h)
    expect_false(isTRUE(all.equal(ggmlR:::.ag_data(got),
                                  ggmlR:::.ag_data(plain), tolerance = 1e-3)))
  })
})

test_that("masked gradients match finite differences", {
  with_cpu_backend({
    d <- 8L; s <- 4L; h <- 2L
    set.seed(33L)
    Q <- matrix(runif(d * s, -1, 1), d, s)
    K <- matrix(runif(d * s, -1, 1), d, s)
    V <- matrix(runif(d * s, -1, 1), d, s)
    G <- matrix(runif(d * s, -1, 1), d, s)
    blocked <- function(j) ifelse(seq_len(s) <= j, 0, -Inf)

    num <- function(which, eps = 1e-5) {
      M <- list(q = Q, k = K, v = V); g <- numeric(d * s)
      for (i in seq_along(g)) {
        hi <- M; hi[[which]][i] <- hi[[which]][i] + eps
        lo <- M; lo[[which]][i] <- lo[[which]][i] - eps
        g[i] <- (sum(G * attn_ref_masked(hi$q, hi$k, hi$v, h, blocked)) -
                 sum(G * attn_ref_masked(lo$q, lo$k, lo$v, h, blocked))) / (2 * eps)
      }
      matrix(g, d, s)
    }

    q <- ag_param(Q); k <- ag_param(K); v <- ag_param(V)
    with_grad_tape({
      out  <- ag_flash_attention(q, k, v, h, causal = TRUE)
      loss <- ag_sum(ag_mul(out, ag_tensor(G)))
    })
    backward(loss)

    # The backward has to receive the SAME mask as the forward; using a
    # different one (or none) gives gradients that look plausible and are wrong.
    expect_equal(q$grad, num("q"), tolerance = 1e-4)
    expect_equal(k$grad, num("k"), tolerance = 1e-4)
    expect_equal(v$grad, num("v"), tolerance = 1e-4)
  })
})

test_that("an explicit mask is accepted in both spellings", {
  with_cpu_backend({
    d <- 8L; sq <- 3L; skv <- 5L; h <- 2L
    set.seed(35L)
    Q <- matrix(runif(d * sq, -1, 1), d, sq)
    K <- matrix(runif(d * skv, -1, 1), d, skv)
    V <- matrix(runif(d * skv, -1, 1), d, skv)

    # Block the last two keys for every query. Mask is [seq_kv, seq_q].
    allow <- matrix(TRUE, skv, sq); allow[(skv - 1L):skv, ] <- FALSE
    numeric_mask <- matrix(0, skv, sq); numeric_mask[(skv - 1L):skv, ] <- -Inf

    a <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h,
                            mask = allow)
    b <- ag_flash_attention(ag_tensor(Q), ag_tensor(K), ag_tensor(V), h,
                            mask = numeric_mask)
    want <- attn_ref_masked(Q, K, V, h,
                            function(j) c(rep(0, skv - 2L), -Inf, -Inf))

    expect_equal(ggmlR:::.ag_data(a), want, tolerance = 1e-5)
    expect_equal(ggmlR:::.ag_data(b), want, tolerance = 1e-5)
  })
})

test_that("a mask in the wrong orientation is rejected", {
  with_cpu_backend({
    d <- 8L; sq <- 3L; skv <- 5L; h <- 2L
    q <- ag_tensor(matrix(0, d, sq))
    k <- ag_tensor(matrix(0, d, skv))
    v <- ag_tensor(matrix(0, d, skv))

    # [seq_q, seq_kv] is the natural way to write it and the wrong way round;
    # rejecting it matters because for a square case it would be accepted
    # silently and mask the transposed entries.
    expect_error(
      ag_flash_attention(q, k, v, h, mask = matrix(0, sq, skv)),
      "seq_kv, seq_q")

    expect_error(
      ag_flash_attention(q, k, v, h, mask = matrix(0, skv, sq), causal = TRUE),
      "either .mask. or .causal.")
  })
})
