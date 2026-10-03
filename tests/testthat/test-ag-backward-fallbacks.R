# Fallbacks from the graph backward to the closures: counted, named, and -- for
# errors, which are bugs rather than refusals -- fatal in strict mode. Plus
# ag_grad(), the one way to read a gradient whatever form $grad has.

skip_if_no_gpu_fb <- function() {
  if (!ggml_vulkan_available() || ggml_vulkan_device_count() < 1L)
    skip("no Vulkan device")
}
fallbacks <- function(reset = FALSE) ggmlR:::ag_backward_fallbacks(reset)

small_tape <- function() {
  set.seed(41)
  W <- ag_param(matrix(rnorm(6), 2, 3))
  x <- ag_tensor(matrix(rnorm(12), 3, 4))
  with_grad_tape({ loss <- ag_mse_loss(ag_matmul(W, x), matrix(0, 2, 4)) })
  list(W = W, x = x, loss = loss)
}

test_that("an f16 tape is refused by name and the refusal is counted", {
  skip_if_no_gpu_fb()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  prev <- ag_dtype("f16"); on.exit(ag_dtype(prev), add = TRUE)
  fallbacks(reset = TRUE)
  tp <- small_tape()
  backward(tp$loss)
  reason <- "closures (compute dtype f16: the graph backward runs in f32 only)"
  expect_identical(ggmlR:::ag_backward_path(), reason)
  expect_identical(fallbacks()[[reason]], 1)
  expect_true(all(is.finite(ag_grad(tp$W))))
})

test_that("strict mode raises an error in the graph path instead of falling back", {
  skip_if_no_gpu_fb()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  local_mocked_bindings(.ag_bwd_run_graph = function(loss, nodes) stop("boom"),
                        .package = "ggmlR")
  fallbacks(reset = TRUE)
  # the run itself may set GGMLR_AG_BWD_STRICT=1; this test drives the option
  withr::local_envvar(GGMLR_AG_BWD_STRICT = "")

  tp <- small_tape()
  withr::local_options(ggmlR.ag_backward_strict = TRUE)
  expect_error(backward(tp$loss), "strict mode, no fallback\\): boom")

  withr::local_options(ggmlR.ag_backward_strict = FALSE)
  tp <- small_tape()
  backward(tp$loss)                                     # falls back, silently...
  expect_identical(ggmlR:::ag_backward_path(), "closures (error: boom)")
  expect_identical(fallbacks()[["closures (error: boom)"]], 1)   # ...but counted
})

test_that("ag_grad() returns a matrix on the closure and the graph path", {
  ag_device("cpu")
  tp <- small_tape(); backward(tp$loss)
  ref <- ag_grad(tp$W)
  expect_true(is.matrix(ref)); expect_identical(dim(ref), c(2L, 3L))
  expect_null(ag_grad(ag_param(matrix(1))))            # no gradient yet
  expect_error(ag_grad(matrix(1)), "must be an ag_tensor")

  skip_if_no_gpu_fb()
  ag_device("gpu"); on.exit(ag_device("cpu"), add = TRUE)
  tp <- small_tape(); backward(tp$loss)
  expect_identical(ggmlR:::ag_backward_path(), "graph")
  g <- ag_grad(tp$W)
  expect_true(is.matrix(g))
  expect_equal(g, ref, tolerance = 1e-5, ignore_attr = TRUE)
})
