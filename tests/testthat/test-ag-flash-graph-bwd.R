# Attention on the graph backward path.
#
# WHAT BROKE, AND WHY IT WAS WORSE THAN IT LOOKED
# ----------------------------------------------
# ag_flash_attention() recorded its tape node without an `op`, and
# .ag_bwd_reject_reason() refuses a tape the moment ANY node lacks one. The
# rejection is all-or-nothing by design (splitting a backward between a graph
# and closures puts the per-op round trip back), so a single attention block
# cost the whole tape its graph backward and its fused forward+backward -- the
# matmuls around it included. The symptom was not an error but a silent fall
# back to closures, which computes correct gradients slowly.
#
# So these tests check two separable things, and the first one is the point:
#
#   1. the tape TAKES the graph path when it holds attention (last_path)
#   2. the gradients are the same ones the closure path produces
#
# A test that only checked (2) would still pass with the bug present, because
# the closure path was always the fallback and was always correct.
#
# The resident path is gpu + f32 only (ggml_flash_attn_back asserts F32 on
# q/k/v/d, and an assert aborts R rather than erroring), so everything here
# skips without a device.

ag_flash_attention <- ggmlR:::ag_flash_attention

bwd_path <- ggmlR:::ag_backward_path

# Run one attention tape and return the leaf gradients as plain matrices,
# forcing either the graph path or the closure path.
run_tape <- function(q, k, v, n_heads, graph, mask = NULL, causal = FALSE,
                     extra = FALSE) {
  old <- ggmlR:::ag_backward_graph(graph)
  withr::defer(ggmlR:::ag_backward_graph(old))
  qq <- ag_param(q); kk <- ag_param(k); vv <- ag_param(v)
  with_grad_tape({
    out <- ag_flash_attention(qq, kk, vv, n_heads = n_heads,
                              mask = mask, causal = causal)
    # `extra` puts an ordinary op after attention, so the test covers the case
    # the bug actually hurt: a tape MIXING attention with ops the graph can
    # emit. With the bug, this tape fell back wholesale.
    if (extra) out <- ag_relu(out)
    loss <- ag_mse_loss(out, ag_tensor(matrix(0, nrow(q), ncol(q))))
  })
  backward(loss)
  list(path = bwd_path(),
       q = ggmlR:::.ag_as_matrix(qq$grad),
       k = ggmlR:::.ag_as_matrix(kk$grad),
       v = ggmlR:::.ag_as_matrix(vv$grad),
       loss = as.numeric(ggmlR:::.ag_as_matrix(ggmlR:::.ag_data(loss))))
}

set.seed(1L)
D <- 8L; H <- 2L; S <- 4L
mk <- function() matrix(runif(D * S, -1, 1), D, S)

test_that("a tape holding attention is no longer rejected by the graph backward", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")

  q <- mk(); k <- mk(); v <- mk()
  res <- run_tape(q, k, v, H, graph = TRUE)

  # The regression itself. Before the fix this read
  # "closures (tape node has no op record)".
  expect_identical(res$path, "graph")
})

test_that("attention mixed with other ops keeps the whole tape on the graph", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")

  q <- mk(); k <- mk(); v <- mk()
  res <- run_tape(q, k, v, H, graph = TRUE, extra = TRUE)
  expect_identical(res$path, "graph")
})

test_that("graph gradients match the closure path", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")

  q <- mk(); k <- mk(); v <- mk()
  g <- run_tape(q, k, v, H, graph = TRUE)
  c_ <- run_tape(q, k, v, H, graph = FALSE)

  expect_identical(g$path, "graph")
  expect_true(startsWith(c_$path, "closures"))

  # Same arithmetic, same device, so this is tighter than a cross-device
  # comparison would justify -- but not exact: the graph reassociates the head
  # permutations through cont/view nodes rather than R aperm.
  expect_equal(g$q, c_$q, tolerance = 1e-4)
  expect_equal(g$k, c_$k, tolerance = 1e-4)
  expect_equal(g$v, c_$v, tolerance = 1e-4)
  expect_equal(g$loss, c_$loss, tolerance = 1e-4)
})

test_that("a causal mask reaches the backward unchanged", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")

  q <- mk(); k <- mk(); v <- mk()
  g <- run_tape(q, k, v, H, graph = TRUE,  causal = TRUE)
  c_ <- run_tape(q, k, v, H, graph = FALSE, causal = TRUE)

  expect_identical(g$path, "graph")
  # The mask is a forward constant; a backward built against a DIFFERENT mask
  # than its forward is wrong in a way nothing reports, so the two paths having
  # to agree is the check that the same mask tensor is reused.
  expect_equal(g$q, c_$q, tolerance = 1e-4)
  expect_equal(g$k, c_$k, tolerance = 1e-4)
  expect_equal(g$v, c_$v, tolerance = 1e-4)
})

test_that("gradients are not silently zero", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f32")

  q <- mk(); k <- mk(); v <- mk()
  g <- run_tape(q, k, v, H, graph = TRUE)

  # A view at a wrong offset in the packed gradient buffer reads the padding,
  # which is never written and therefore reads as zero. dq sits at offset 0 and
  # would survive that; dk and dv are the ones that shift, which is why all
  # three are checked rather than the sum.
  expect_gt(max(abs(g$q)), 0)
  expect_gt(max(abs(g$k)), 0)
  expect_gt(max(abs(g$v)), 0)
  expect_true(all(is.finite(g$q)))
  expect_true(all(is.finite(g$k)))
  expect_true(all(is.finite(g$v)))
})

test_that("f16 computes an attention gradient instead of aborting R", {
  skip_if_not(ggml_vulkan_available(), "Vulkan not available")
  skip_if_not(ggml_vulkan_device_count() > 0, "No Vulkan devices")
  local_cpu_device()
  ag_device("gpu"); ag_dtype("f16")
  withr::defer(ag_dtype("f32"))

  # REGRESSION, and it predates the graph backward entirely.
  #
  # ggml_flash_attn_back asserts F32 on q/k/v/d (ggml-ops-builders.c:3701).
  # .ag_flash_run built its tensors from .ag_compute_dtype(), so ag_dtype("f16")
  # plus any backward through attention hit that assert -- and a GGML_ASSERT is
  # an abort, not an R condition: the session died with no traceback and nothing
  # to catch. The forward alone was fine, which is why it went unnoticed.
  #
  # Two things are checked here, and the first one is the whole point:
  #   1. this returns at all
  #   2. the numbers are finite and non-zero
  # The resident path declines f16 up front (.ag_flash_resident_ok) and the
  # closure path now forces F32 for the backward, so both routes survive.
  q <- mk(); k <- mk(); v <- mk()
  res <- expect_no_error(run_tape(q, k, v, H, graph = TRUE))

  expect_true(startsWith(res$path, "closures"))
  expect_true(all(is.finite(res$q)))
  expect_true(all(is.finite(res$k)))
  expect_true(all(is.finite(res$v)))
  expect_gt(max(abs(res$q)), 0)
  expect_gt(max(abs(res$k)), 0)
  expect_gt(max(abs(res$v)), 0)
})
