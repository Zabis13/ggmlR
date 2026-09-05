# Tests for the ag_* residency infrastructure (memory side).
#
# Three independent risks are covered here, each with its own test:
#   1. buffer leak      — every alloc_ctx_tensors() call returns a NEW buffer
#                         for the newly created tensors only; keeping just the
#                         last one loses every earlier buffer.
#   2. context growth   — a context that runs out of descriptor space aborts R
#                         inside ggml_new_tensor_impl (GGML_ASSERT(obj_new)),
#                         so the overflow must be predicted BEFORE the tensor
#                         is created and a fresh context started instead.
#   3. stale $ptr       — resetting the context frees the backend buffers, but
#                         an ag_tensor outlives the context and keeps its $ptr.
#                         Reading it afterwards is a use-after-free that returns
#                         plausible garbage rather than failing loudly.

# Residency lives on the device-state backend. These tests exercise the memory
# bookkeeping only, so any backend will do — the CPU one keeps them runnable
# without a GPU.
.ag_test_backend <- function() {
  if (is.null(.ag_device_state$backend)) {
    .ag_device_state$backend <- ggml_backend_cpu_init()
  }
  invisible(.ag_device_state$backend)
}

test_that("ag residency state starts clean", {
  .ag_residency_reset()
  expect_length(.ag_device_state$buffers, 0L)
  expect_null(.ag_device_state$ctx)
  expect_type(.ag_device_state$ctx_gen, "integer")
})

test_that("context generation advances on every reset", {
  .ag_residency_reset()
  g0 <- .ag_device_state$ctx_gen
  .ag_residency_reset()
  g1 <- .ag_device_state$ctx_gen
  .ag_residency_reset()
  g2 <- .ag_device_state$ctx_gen

  expect_gt(g1, g0)
  expect_gt(g2, g1)
})

# ---------------------------------------------------------------------------
# Risk 1: buffer leak
# ---------------------------------------------------------------------------

test_that("every allocated buffer is retained, not overwritten", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()
  .ag_ctx_ensure()

  # Upload several tensors one at a time. Each upload creates a tensor and
  # allocates it, which yields a fresh buffer covering only that tensor.
  n <- 5L
  for (i in seq_len(n)) .ag_r_to_gpu(matrix(as.numeric(i), 4L, 4L))

  # The leak this guards against: keeping a single $buffer slot means only the
  # last allocation is reachable and the earlier ones can never be freed.
  expect_gt(length(.ag_device_state$buffers), 0L)

  # Accounted memory must cover all uploads, so it has to grow with n.
  total <- .ag_tape_mem()$buffer_bytes
  expect_gt(total, 0)

  .ag_r_to_gpu(matrix(0, 64L, 64L))
  expect_gt(.ag_tape_mem()$buffer_bytes, total)

  .ag_residency_reset()
  expect_length(.ag_device_state$buffers, 0L)
})

test_that("reset frees buffers and drops them from the ledger", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()
  .ag_ctx_ensure()
  .ag_r_to_gpu(matrix(1, 8L, 8L))
  expect_gt(.ag_tape_mem()$buffer_bytes, 0)

  .ag_residency_reset()
  expect_identical(.ag_tape_mem()$buffer_bytes, 0)
  expect_length(.ag_device_state$buffers, 0L)
})

test_that("a batch upload takes one buffer, not one per tensor", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  # This is the point of batching: Vulkan limits how MANY allocations a device
  # will grant (maxMemoryAllocationCount, commonly 4096), so a long tape that
  # allocated per tensor would fail with VK_ERROR_TOO_MANY_OBJECTS.
  .ag_residency_reset()
  .ag_ctx_ensure()

  mats <- replicate(16L, matrix(runif(16L), 4L, 4L), simplify = FALSE)
  ptrs <- .ag_r_to_gpu_batch(mats)

  expect_length(ptrs, length(mats))
  expect_identical(.ag_tape_mem()$n_buffers, 1L)

  # Batching must not corrupt the mapping: each tensor reads back its own value.
  for (i in seq_along(mats)) {
    back <- matrix(ggml_backend_tensor_get_data(ptrs[[i]]), 4L, 4L)
    expect_equal(back, mats[[i]], tolerance = 1e-6)
  }

  .ag_residency_reset()
})

# ---------------------------------------------------------------------------
# Risk 2: context growth
# ---------------------------------------------------------------------------

test_that("context overflow is predicted, not hit", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  # A context sized for only a handful of descriptors. Creating more tensors
  # than fit must roll over to a new context rather than abort R.
  .ag_residency_reset(size_mb = .ag_min_ctx_mb())
  .ag_ctx_ensure()
  first_ctx <- .ag_device_state$ctx

  # Enough tensors to exceed the tiny context several times over.
  n_fit <- .ag_ctx_capacity(first_ctx)
  expect_gt(n_fit, 0)

  for (i in seq_len(n_fit * 3L)) .ag_r_to_gpu(matrix(0, 2L, 2L))

  # Survived without abort, and rolled over to further contexts.
  expect_gt(length(.ag_device_state$contexts), 1L)

  # Old contexts stay alive: their tensors must remain readable.
  expect_true(all(vapply(.ag_device_state$contexts,
                         function(c) !is.null(c), logical(1))))

  .ag_residency_reset()
})

test_that("tensors created before a rollover stay valid after it", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset(size_mb = .ag_min_ctx_mb())
  .ag_ctx_ensure()

  early <- matrix(c(1, 2, 3, 4), 2L, 2L)
  ptr   <- .ag_r_to_gpu(early)
  gen   <- .ag_device_state$ctx_gen

  # Force a rollover into a new context.
  n_fit <- .ag_ctx_capacity(.ag_device_state$ctx)
  for (i in seq_len(n_fit * 2L)) .ag_r_to_gpu(matrix(0, 2L, 2L))
  expect_gt(length(.ag_device_state$contexts), 1L)

  # A rollover must NOT invalidate earlier tensors: the generation is unchanged
  # (only a reset bumps it) and the data reads back intact.
  expect_identical(.ag_device_state$ctx_gen, gen)
  back <- matrix(ggml_backend_tensor_get_data(ptr), 2L, 2L)
  expect_equal(back, early, tolerance = 1e-6)

  .ag_residency_reset()
})

# ---------------------------------------------------------------------------
# Risk 3: stale $ptr after reset (use-after-free)
# ---------------------------------------------------------------------------

test_that(".ag_data ignores a pointer from a freed context", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()
  .ag_ctx_ensure()

  m <- matrix(c(1, 2, 3, 4), 2L, 2L)
  t <- ag_tensor(m, device = "gpu")
  t$ptr     <- .ag_r_to_gpu(m)
  t$shape   <- dim(m)
  t$ctx_gen <- .ag_device_state$ctx_gen

  # While the generation matches, the pointer is the source of truth.
  expect_equal(.ag_data(t), m, tolerance = 1e-6)

  # After a reset the buffer behind $ptr is freed. Reading it would be a
  # use-after-free returning plausible garbage, so the stale pointer must be
  # ignored in favour of the retained R matrix.
  .ag_residency_reset()
  expect_equal(.ag_data(t), m, tolerance = 1e-6)

  .ag_residency_reset()
})

test_that("a stale pointer with no fallback data is a clear error", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()
  .ag_ctx_ensure()

  m <- matrix(c(1, 2, 3, 4), 2L, 2L)
  t <- ag_tensor(m, device = "gpu")
  t$ptr     <- .ag_r_to_gpu(m)
  t$shape   <- dim(m)
  t$ctx_gen <- .ag_device_state$ctx_gen
  t$data    <- NULL          # resident-only tensor, nothing to fall back to

  .ag_residency_reset()

  # Must fail loudly rather than read freed memory.
  expect_error(.ag_data(t), "freed|stale|generation", ignore.case = TRUE)

  .ag_residency_reset()
})

test_that("ag_tensor carries residency fields", {
  t <- ag_tensor(matrix(1, 2L, 2L))
  expect_true(is.null(t$ptr))
  expect_true(is.null(t$ctx_gen))
  # $shape is only meaningful once the tensor is resident; it must not shadow
  # the R matrix dimensions before that.
  expect_equal(dim(t$data), c(2L, 2L))
})

# ---------------------------------------------------------------------------
# Memory ledger
# ---------------------------------------------------------------------------

test_that("tape memory ledger reports context and buffer usage", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()
  .ag_ctx_ensure()

  m0 <- .ag_tape_mem()
  # The unprefixed fields describe the pass pool, as they always have; the p_*
  # fields and the total are what the persistent pool added.
  expect_named(m0, c("ctx_bytes", "ctx_used", "buffer_bytes", "n_contexts",
                     "n_buffers",
                     "p_ctx_bytes", "p_ctx_used", "p_buffer_bytes",
                     "p_n_contexts", "p_n_buffers", "total_buffer_bytes"),
               ignore.order = TRUE)
  expect_identical(m0$buffer_bytes, 0)

  .ag_r_to_gpu(matrix(0, 32L, 32L))
  m1 <- .ag_tape_mem()
  expect_gt(m1$buffer_bytes, m0$buffer_bytes)
  expect_gt(m1$ctx_used, 0)

  .ag_residency_reset()
})

test_that("tape memory limit raises before the driver runs out", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()
  .ag_ctx_ensure()

  old <- .ag_tape_mem_limit(1024)          # 1 KB budget
  on.exit(.ag_tape_mem_limit(old), add = TRUE)

  expect_error(.ag_r_to_gpu(matrix(0, 256L, 256L)), "budget|limit|memory",
               ignore.case = TRUE)

  .ag_residency_reset()
})

# ---------------------------------------------------------------------------
# Risk 4: two contexts with different lifetimes in one op
# ---------------------------------------------------------------------------
#
# .ag_run_op keeps tensors in the persistent residency context but builds each
# op's graph in a throwaway context that is freed as the call returns. The
# graph's nodes[]/leafs[] arrays hold pointers to tensors owned by the OTHER
# context, so freeing the graph context must not disturb them.

test_that("freeing the per-op graph context leaves resident tensors intact", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()

  a <- matrix(c(1, 2, 3, 4), 2L, 2L)
  ptr <- .ag_r_to_gpu(a)
  gen <- .ag_device_state$ctx_gen

  # Each of these builds and frees a graph context of its own.
  for (i in 1:5) .ag_gpu_add(a, a)

  # The tensor uploaded before those ops still reads back its own value: the
  # graph contexts died without touching the residency context.
  expect_identical(.ag_device_state$ctx_gen, gen)
  expect_equal(matrix(ggml_backend_tensor_get_data(ptr), 2L, 2L), a,
               tolerance = 1e-6)

  .ag_residency_reset()
})

test_that("ops reuse the residency context instead of one per call", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()
  .ag_ctx_ensure()
  n_ctx0 <- length(.ag_device_state$contexts)

  a <- matrix(runif(64L), 8L, 8L)
  for (i in 1:8) .ag_gpu_add(a, a)

  # Tensors accumulate in the shared context; a rollover is allowed once it
  # fills, but nothing like one context per op.
  expect_lt(length(.ag_device_state$contexts), n_ctx0 + 8L)

  .ag_residency_reset()
})

test_that("op results are still correct through the shared context", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()

  a <- matrix(c(1, 2, 3, 4), 2L, 2L)
  b <- matrix(c(5, 6, 7, 8), 2L, 2L)

  expect_equal(.ag_gpu_add(a, b), a + b, tolerance = 1e-5)
  expect_equal(.ag_gpu_mul(a, b), a * b, tolerance = 1e-5)
  expect_equal(.ag_gpu_matmul(a, b), a %*% b, tolerance = 1e-5)

  # Repeat: the second run goes through a context that already holds tensors.
  expect_equal(.ag_gpu_add(a, b), a + b, tolerance = 1e-5)
  expect_equal(.ag_gpu_matmul(a, b), a %*% b, tolerance = 1e-5)

  .ag_residency_reset()
})

# ---------------------------------------------------------------------------
# Risk 5: the value-access contract (inst/docs/ag_data_contract.md)
# ---------------------------------------------------------------------------

test_that(".ag_data_set drops device residency so the next read is not stale", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()

  t <- ag_tensor(matrix(c(1, 2, 3, 4), 2L, 2L), device = "gpu")
  t$ptr     <- .ag_r_to_gpu(t$data)
  t$shape   <- c(2L, 2L)
  t$ctx_gen <- .ag_device_state$ctx_gen

  expect_equal(.ag_data(t), matrix(c(1, 2, 3, 4), 2L, 2L), tolerance = 1e-6)

  # Writing a new value must invalidate the pointer, not leave the device
  # holding the old one.
  new_val <- matrix(c(9, 9, 9, 9), 2L, 2L)
  .ag_data_set(t, new_val)

  expect_null(t$ptr)
  expect_null(t$ctx_gen)
  expect_equal(.ag_data(t), new_val, tolerance = 1e-6)

  .ag_residency_reset()
})

test_that(".ag_data materialisation cache does not outlive its generation", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()

  t <- ag_tensor(matrix(c(1, 2, 3, 4), 2L, 2L), device = "gpu")
  t$ptr     <- .ag_r_to_gpu(t$data)
  t$shape   <- c(2L, 2L)
  t$ctx_gen <- .ag_device_state$ctx_gen

  expect_equal(.ag_data(t), matrix(c(1, 2, 3, 4), 2L, 2L), tolerance = 1e-6)

  # After a reset the pointer belongs to a dead generation. The cached download
  # must not be presented as if it were still device-backed: the fallback is
  # the retained matrix, and the generation no longer matches.
  .ag_residency_reset()
  expect_false(.ag_ptr_is_live(t))
  expect_equal(.ag_data(t), matrix(c(1, 2, 3, 4), 2L, 2L), tolerance = 1e-6)

  .ag_residency_reset()
})

test_that(".ag_data_mut refuses a tensor with no obtainable value", {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  .ag_test_backend()

  .ag_residency_reset()

  t <- ag_tensor(matrix(0, 2L, 2L), device = "gpu")
  t$ptr     <- .ag_r_to_gpu(t$data)
  t$shape   <- c(2L, 2L)
  t$ctx_gen <- .ag_device_state$ctx_gen
  t$data    <- NULL          # no host fallback left

  .ag_residency_reset()      # pointer now stale

  expect_error(.ag_data_mut(t), "stale|freed|unavailable", ignore.case = TRUE)

  .ag_residency_reset()
})

test_that("optimizers update parameters through the contract", {
  .ag_residency_reset()

  p <- ag_param(matrix(c(1, 1, 1, 1), 2L, 2L))
  before <- .ag_data(p)

  opt <- optimizer_sgd(list(w = p), lr = 0.1)
  grads <- new.env(parent = emptyenv())
  assign(as.character(p$id), matrix(1, 2L, 2L), envir = grads)
  opt$step(grads)

  # Value moved by exactly -lr * grad, and it is visible through .ag_data().
  expect_equal(.ag_data(p), before - 0.1, tolerance = 1e-6)
})

test_that("device ops fail cleanly when no backend was initialised", {
  # ag_device("cpu") only records the choice; it creates no backend. Reaching a
  # device op in that state used to pass an R NULL into
  # R_ggml_backend_alloc_ctx_tensors, where R_ExternalPtrAddr() turned it into a
  # garbage address that survived the != NULL check and segfaulted in
  # ggml-alloc. Unreachable through the public ops (they dispatch to .ag_gpu_*
  # only when device == "gpu"), but a crash is never an acceptable failure mode.
  old_backend <- .ag_device_state$backend
  old_device  <- .ag_device_state$device
  on.exit({
    .ag_device_state$backend <- old_backend
    .ag_device_state$device  <- old_device
  }, add = TRUE)

  .ag_residency_reset()
  .ag_device_state$backend <- NULL

  expect_error(.ag_gpu_matmul(matrix(1, 4L, 4L), matrix(1, 4L, 4L)),
               "backend", ignore.case = TRUE)

  .ag_device_state$backend <- old_backend
  .ag_residency_reset()
})
