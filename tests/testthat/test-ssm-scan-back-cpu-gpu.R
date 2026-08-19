# Backward of ggml_ssm_scan on the CPU and on Vulkan.
#
# ggmlR extension: ggml has no Vulkan shader for SSM_SCAN_BACK. This was the
# expensive half of training a Mamba block -- with the backward on the CPU a
# training step cost ~22ms against ~0.6ms for the forward on the GPU, so the
# GPU forward bought nothing.
#
# The shader is checked against the CPU kernel AND against a numeric gradient,
# because two backends wrong in the same way would agree with each other.

# Sizes the Vulkan shader accepts: d_state must be 128 or 256, and the scan
# must be Mamba-2 (A of shape [1, n_head]).
sb_dims <- function(d_state = 128L, head_dim = 4L, n_head = 2L,
                    n_tok = 3L, n_seqs = 1L) {
  list(d_state = d_state, head_dim = head_dim, n_head = n_head,
       n_tok = n_tok, n_seqs = n_seqs)
}

sb_inputs <- function(d, seed = 5L) {
  set.seed(seed)
  list(
    s0  = runif(d$d_state * d$head_dim * d$n_head * d$n_seqs, -0.5, 0.5),
    x   = runif(d$head_dim * d$n_head * d$n_tok * d$n_seqs, -1, 1),
    dt  = runif(d$n_head * d$n_tok * d$n_seqs, -0.5, 0.5),
    A   = -runif(d$n_head, 0.2, 1.0),
    B   = runif(d$d_state * d$n_tok * d$n_seqs, -0.5, 0.5),
    C   = runif(d$d_state * d$n_tok * d$n_seqs, -0.5, 0.5),
    # Gradient w.r.t. the packed forward result: outputs then final states.
    g   = runif(d$head_dim * d$n_head * d$n_tok * d$n_seqs +
                d$d_state * d$head_dim * d$n_head * d$n_seqs, -1, 1)
  )
}

# Which backend the scheduler assigned a node to, as a plain string.
#
# ggml_vulkan_backend_name() errors with "Vulkan support not compiled" on a
# build without Vulkan, even when asked about a CPU backend, so the CPU-only
# tests below cannot call it unguarded.
sched_backend_name <- function(sched, tensor) {
  if (!ggml_vulkan_available()) {
    return("CPU")
  }
  b <- ggml_backend_sched_get_tensor_backend(sched, tensor)
  if (is.null(b)) "unassigned" else ggml_vulkan_backend_name(b)
}

# Build the backward graph once and run it on the requested backend.
run_scan_back <- function(d, inp, gpu) {
  ctx <- ggml_init(512 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  t_s  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$d_state, d$head_dim,
                             d$n_head, d$n_seqs)
  t_x  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$head_dim, d$n_head,
                             d$n_tok, d$n_seqs)
  t_dt <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d$n_head, d$n_tok, d$n_seqs)
  t_A  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, d$n_head)
  t_B  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$d_state, 1L, d$n_tok, d$n_seqs)
  t_C  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$d_state, 1L, d$n_tok, d$n_seqs)
  t_id <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, d$n_seqs)
  t_g  <- ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length(inp$g))

  out <- ggml_ssm_scan_back(ctx, t_s, t_x, t_dt, t_A, t_B, t_C, t_id, t_g)
  ggml_set_output(out)
  graph <- ggml_build_forward_expand(ctx, out)

  backend <- if (gpu) ggml_vulkan_init(0L) else ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)

  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)

  ggml_backend_tensor_set_data(t_s,  inp$s0)
  ggml_backend_tensor_set_data(t_x,  inp$x)
  ggml_backend_tensor_set_data(t_dt, inp$dt)
  ggml_backend_tensor_set_data(t_A,  inp$A)
  ggml_backend_tensor_set_data(t_B,  inp$B)
  ggml_backend_tensor_set_data(t_C,  inp$C)
  ggml_backend_tensor_set_data(t_id, as.integer(seq_len(d$n_seqs) - 1L))
  ggml_backend_tensor_set_data(t_g,  inp$g)

  ggml_backend_sched_graph_compute(sched, graph)

  list(data  = ggml_backend_tensor_get_data(out),
       where = sched_backend_name(sched, out))
}

# Split the packed result into its six gradients, in the order ggml_ssm_scan
# takes its inputs.
sb_split <- function(v, d) {
  n_s  <- d$d_state * d$head_dim * d$n_head * d$n_seqs
  n_x  <- d$head_dim * d$n_head * d$n_tok * d$n_seqs
  n_dt <- d$n_head * d$n_tok * d$n_seqs
  n_A  <- d$n_head
  n_B  <- d$d_state * d$n_tok * d$n_seqs
  o <- 0
  take <- function(n) { r <- v[o + seq_len(n)]; o <<- o + n; r }
  list(d_s = take(n_s), d_x = take(n_x), d_dt = take(n_dt),
       d_A = take(n_A), d_B = take(n_B), d_C = take(n_B))
}

# Scalar loss whose gradient the op should produce: sum(forward_result * g).
# Running the forward scan on the CPU and dotting with g gives a function of the
# parameters that can be differentiated numerically.
sb_loss <- function(d, inp, par_name, par_value) {
  inp[[par_name]] <- par_value
  ctx <- ggml_init(256 * 1024 * 1024)
  on.exit(ggml_free(ctx), add = TRUE)
  ggml_set_no_alloc(ctx, TRUE)

  t_s  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$d_state, d$head_dim,
                             d$n_head, d$n_seqs)
  t_x  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$head_dim, d$n_head,
                             d$n_tok, d$n_seqs)
  t_dt <- ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d$n_head, d$n_tok, d$n_seqs)
  t_A  <- ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1L, d$n_head)
  t_B  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$d_state, 1L, d$n_tok, d$n_seqs)
  t_C  <- ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d$d_state, 1L, d$n_tok, d$n_seqs)
  t_id <- ggml_new_tensor_1d(ctx, GGML_TYPE_I32, d$n_seqs)

  out <- ggml_ssm_scan(ctx, t_s, t_x, t_dt, t_A, t_B, t_C, t_id)
  ggml_set_output(out)
  graph <- ggml_build_forward_expand(ctx, out)

  backend <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(backend), add = TRUE)
  sched <- ggml_backend_sched_new(list(backend), parallel = FALSE)
  on.exit(ggml_backend_sched_free(sched), add = TRUE)
  ggml_backend_sched_reset(sched)
  ggml_backend_sched_alloc_graph(sched, graph)

  ggml_backend_tensor_set_data(t_s,  inp$s0)
  ggml_backend_tensor_set_data(t_x,  inp$x)
  ggml_backend_tensor_set_data(t_dt, inp$dt)
  ggml_backend_tensor_set_data(t_A,  inp$A)
  ggml_backend_tensor_set_data(t_B,  inp$B)
  ggml_backend_tensor_set_data(t_C,  inp$C)
  ggml_backend_tensor_set_data(t_id, as.integer(seq_len(d$n_seqs) - 1L))

  ggml_backend_sched_graph_compute(sched, graph)
  sum(ggml_backend_tensor_get_data(out) * inp$g)
}

# Central-difference gradient of sb_loss w.r.t. one parameter vector.
sb_numeric_grad <- function(d, inp, par_name, idx, h = 1e-3) {
  base <- inp[[par_name]]
  vapply(idx, function(i) {
    up <- base; up[i] <- up[i] + h
    dn <- base; dn[i] <- dn[i] - h
    (sb_loss(d, inp, par_name, up) - sb_loss(d, inp, par_name, dn)) / (2 * h)
  }, numeric(1))
}

test_that("ssm_scan_back on the CPU matches a numeric gradient", {
  d   <- sb_dims()
  inp <- sb_inputs(d)
  got <- sb_split(run_scan_back(d, inp, gpu = FALSE)$data, d)

  # A sample of indices rather than all of them: each numeric gradient costs two
  # full forward scans, and a wrong kernel is wrong everywhere.
  for (nm in c("dt", "A", "B", "C", "x")) {
    key <- switch(nm, dt = "d_dt", A = "d_A", B = "d_B", C = "d_C", x = "d_x")
    n   <- length(inp[[nm]])
    idx <- unique(round(seq(1, n, length.out = min(6, n))))
    expect_equal(got[[key]][idx], sb_numeric_grad(d, inp, nm, idx),
                 tolerance = 1e-2, info = paste("gradient w.r.t.", nm))
  }
})

test_that("ssm_scan_back on Vulkan matches the CPU kernel", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0, "No Vulkan devices found")

  d   <- sb_dims()
  inp <- sb_inputs(d)

  cpu <- run_scan_back(d, inp, gpu = FALSE)
  gpu <- run_scan_back(d, inp, gpu = TRUE)

  # Without this the comparison could pass while the op quietly ran on the CPU.
  expect_equal(gpu$where, "Vulkan0")
  expect_equal(gpu$data, cpu$data, tolerance = 1e-4)
})

test_that("ssm_scan_back agrees between CPU and Vulkan across shapes", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0, "No Vulkan devices found")

  shapes <- list(
    sb_dims(d_state = 128L, head_dim = 1L,  n_head = 1L, n_tok = 1L, n_seqs = 1L),
    sb_dims(d_state = 128L, head_dim = 8L,  n_head = 4L, n_tok = 7L, n_seqs = 1L),
    sb_dims(d_state = 256L, head_dim = 4L,  n_head = 2L, n_tok = 5L, n_seqs = 1L),
    sb_dims(d_state = 128L, head_dim = 4L,  n_head = 2L, n_tok = 3L, n_seqs = 3L),
    sb_dims(d_state = 128L, head_dim = 64L, n_head = 8L, n_tok = 16L, n_seqs = 1L)
  )

  for (d in shapes) {
    inp <- sb_inputs(d, seed = d$head_dim * 31L + d$n_tok)
    cpu <- run_scan_back(d, inp, gpu = FALSE)
    gpu <- run_scan_back(d, inp, gpu = TRUE)

    lbl <- sprintf("d_state=%d head_dim=%d n_head=%d n_tok=%d n_seqs=%d",
                   d$d_state, d$head_dim, d$n_head, d$n_tok, d$n_seqs)
    expect_equal(gpu$where, "Vulkan0", info = lbl)
    expect_equal(gpu$data, cpu$data, tolerance = 1e-4, info = lbl)
  }
})

test_that("an unsupported d_state falls back to the CPU", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0, "No Vulkan devices found")

  # Mamba-1's d_state=16 is outside what the shader handles, so supports_op
  # must refuse it -- and the result must still be right.
  d   <- sb_dims(d_state = 16L)
  inp <- sb_inputs(d)
  gpu <- run_scan_back(d, inp, gpu = TRUE)
  cpu <- run_scan_back(d, inp, gpu = FALSE)

  expect_equal(gpu$where, "CPU")
  expect_equal(gpu$data, cpu$data, tolerance = 1e-5)
})
