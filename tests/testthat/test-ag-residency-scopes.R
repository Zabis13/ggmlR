# Tests for the two residency lifetimes ("pass" and "persistent").
#
# What this is for. with_grad_tape() begins every tape by freeing the residency
# contexts, which happens once per training step. While there was a single pool,
# that made resident weights impossible: anything left on the device was
# destroyed at the next step, and the weights were re-uploaded every time —
# measured at 74-98% of a forward pass (inst/scripts/proto_ag_weight_cache.R).
#
# Splitting the pools is what removes that. The properties worth pinning down
# are therefore:
#   1. a pass reset frees pass memory and leaves the persistent pool alone;
#   2. the generation counters are independent, so freeing one pool cannot make
#      the other's handles look stale — nor make a dead handle look live;
#   3. a bare .ag_residency_reset() still frees everything, because every
#      existing caller means "clean slate" by it;
#   4. the memory ledger and the VRAM budget see both pools.

# Everything under test is internal, so it is reached through the namespace
# rather than by bare name: that way the file runs the same under test_file()
# as under test_check(), which is the only reason the bare-name style works
# elsewhere.
ns        <- asNamespace("ggmlR")
dev_state <- get(".ag_device_state",     envir = ns)
reset     <- get(".ag_residency_reset",  envir = ns)
ctx_ens   <- get(".ag_ctx_ensure",       envir = ns)
ctx_flush <- get(".ag_ctx_flush",        envir = ns)
tape_mem  <- get(".ag_tape_mem",         envir = ns)
mem_limit <- get(".ag_tape_mem_limit",   envir = ns)
chk_budget<- get(".ag_check_mem_budget", envir = ns)
r_to_gpu  <- get(".ag_r_to_gpu",         envir = ns)
mk_handle <- get(".ag_handle",           envir = ns)
h_live    <- get(".ag_handle_live",      envir = ns)
h_scope   <- get(".ag_handle_scope",     envir = ns)
t_scope   <- get(".ag_tensor_scope",     envir = ns)
ptr_live  <- get(".ag_ptr_is_live",      envir = ns)
set_handle<- get(".ag_data_set_handle",  envir = ns)
ag_data   <- get(".ag_data",             envir = ns)

# Residency lives on the device-state backend. These tests exercise the memory
# bookkeeping only, so any backend will do — the CPU one keeps them runnable
# without a GPU.
.ag_scope_test_backend <- function() {
  skip_if(ggml_backend_dev_count() < 1, "No ggml backend device available")
  if (is.null(dev_state$backend)) {
    dev_state$backend <- ggml_backend_cpu_init()
  }
  invisible(dev_state$backend)
}

test_that("both pools start clean and independent", {
  reset()

  expect_length(dev_state$contexts, 0L)
  expect_length(dev_state$p_contexts, 0L)
  expect_null(dev_state$ctx)
  expect_null(dev_state$p_ctx)
  expect_type(dev_state$ctx_gen, "integer")
  expect_type(dev_state$p_ctx_gen, "integer")
})

test_that("a pass reset leaves the persistent pool untouched", {
  .ag_scope_test_backend()
  reset()

  r_to_gpu(matrix(0, 32L, 32L), scope = "persistent")
  r_to_gpu(matrix(0, 32L, 32L), scope = "pass")

  before <- tape_mem()
  expect_gt(before$buffer_bytes, 0)
  expect_gt(before$p_buffer_bytes, 0)

  reset(scope = "pass")

  after <- tape_mem()
  # The pass pool is gone; the persistent one is byte-for-byte what it was.
  expect_identical(after$buffer_bytes, 0)
  expect_identical(after$p_buffer_bytes, before$p_buffer_bytes)
  expect_gt(after$p_n_buffers, 0L)

  reset()
})

test_that("a persistent reset leaves the pass pool untouched", {
  .ag_scope_test_backend()
  reset()

  r_to_gpu(matrix(0, 32L, 32L), scope = "persistent")
  r_to_gpu(matrix(0, 32L, 32L), scope = "pass")
  before <- tape_mem()

  reset(scope = "persistent")

  after <- tape_mem()
  expect_identical(after$p_buffer_bytes, 0)
  expect_identical(after$buffer_bytes, before$buffer_bytes)

  reset()
})

test_that("generation counters advance independently", {
  reset()
  g0 <- dev_state$ctx_gen
  p0 <- dev_state$p_ctx_gen

  reset(scope = "pass")
  expect_identical(dev_state$ctx_gen, g0 + 1L)
  expect_identical(dev_state$p_ctx_gen, p0)      # persistent stood still

  reset(scope = "persistent")
  expect_identical(dev_state$ctx_gen, g0 + 1L)   # pass stood still
  expect_identical(dev_state$p_ctx_gen, p0 + 1L)

  reset()                                  # "all" bumps both
  expect_identical(dev_state$ctx_gen, g0 + 2L)
  expect_identical(dev_state$p_ctx_gen, p0 + 2L)
})

test_that("a bare reset still frees everything", {
  .ag_scope_test_backend()
  reset()

  r_to_gpu(matrix(0, 32L, 32L), scope = "persistent")
  r_to_gpu(matrix(0, 32L, 32L), scope = "pass")
  expect_gt(tape_mem()$total_buffer_bytes, 0)

  # The default is "all" precisely so that existing callers — tests, ag_device(),
  # anything that wants a clean slate — keep the behaviour they were written for.
  reset()

  m <- tape_mem()
  expect_identical(m$buffer_bytes, 0)
  expect_identical(m$p_buffer_bytes, 0)
  expect_identical(m$total_buffer_bytes, 0)
})

test_that("a persistent handle survives a pass reset; a pass handle does not", {
  .ag_scope_test_backend()
  reset()

  ctx_p <- ctx_ens(1L, scope = "persistent")
  t_p   <- ggml_new_tensor_2d(ctx_p, GGML_TYPE_F32, 4L, 4L)
  ctx_flush(ctx_p, scope = "persistent")
  h_p   <- mk_handle(t_p, c(4L, 4L), scope = "persistent")

  ctx_a <- ctx_ens(1L, scope = "pass")
  t_a   <- ggml_new_tensor_2d(ctx_a, GGML_TYPE_F32, 4L, 4L)
  ctx_flush(ctx_a, scope = "pass")
  h_a   <- mk_handle(t_a, c(4L, 4L), scope = "pass")

  expect_true(h_live(h_p))
  expect_true(h_live(h_a))

  reset(scope = "pass")

  # This is the whole point of the split: the weight is still addressable after
  # the event that happens once per training step, and the activation is not.
  expect_true(h_live(h_p))
  expect_false(h_live(h_a))

  reset()
  expect_false(h_live(h_p))
})

test_that("a handle is checked against its own pool, not the other one", {
  reset()

  # Bump the pass counter well past the persistent one. A handle checked
  # against the wrong counter would now be declared stale while its memory is
  # perfectly alive — the failure mode the scope field exists to prevent.
  for (i in seq_len(3L)) reset(scope = "pass")

  .ag_scope_test_backend()
  ctx_p <- ctx_ens(1L, scope = "persistent")
  t_p   <- ggml_new_tensor_2d(ctx_p, GGML_TYPE_F32, 4L, 4L)
  ctx_flush(ctx_p, scope = "persistent")
  h_p   <- mk_handle(t_p, c(4L, 4L), scope = "persistent")

  expect_false(identical(dev_state$ctx_gen, dev_state$p_ctx_gen))
  expect_true(h_live(h_p))

  reset()
})

test_that("handles default to the pass pool", {
  reset()
  h <- mk_handle(NULL, c(2L, 2L))
  expect_identical(h_scope(h), "pass")
  expect_identical(h$gen, dev_state$ctx_gen)
})

test_that("the memory budget counts both pools", {
  .ag_scope_test_backend()
  reset()

  # A resident weight occupies device memory exactly as an activation does, so
  # a budget blind to the persistent pool would let VRAM fill up unreported.
  r_to_gpu(matrix(0, 64L, 64L), scope = "persistent")
  held <- tape_mem()$total_buffer_bytes
  expect_gt(held, 0)

  old <- mem_limit(held / 2)
  on.exit({
    mem_limit(old)
    reset()
  }, add = TRUE)

  expect_error(chk_budget(1024), "budget exceeded")
})

test_that("a resident tensor is validated against its own pool", {
  .ag_scope_test_backend()
  reset()

  ctx_p <- ctx_ens(1L, scope = "persistent")
  ptr   <- ggml_new_tensor_2d(ctx_p, GGML_TYPE_F32, 2L, 2L)
  ctx_flush(ctx_p, scope = "persistent")
  ggml_backend_tensor_set_data(ptr, as.numeric(matrix(1:4, 2L, 2L)))

  t <- ag_tensor(matrix(0, 2L, 2L))
  t$device <- "gpu"
  set_handle(t, mk_handle(ptr, c(2L, 2L), scope = "persistent"))

  expect_identical(t_scope(t), "persistent")
  expect_true(ptr_live(t))

  # The tape reset that runs every training step must not invalidate it.
  reset(scope = "pass")
  expect_true(ptr_live(t))
  expect_equal(ag_data(t), matrix(1:4, 2L, 2L), tolerance = 1e-6)

  reset()
})
