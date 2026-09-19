# Differential backend testing: every op ggmlR builds must agree with the CPU.
#
# ggml_test_backend_ops() runs each case on the CPU and on the accelerated
# backend with identical inputs and reports the normalized mean squared error.
# Each case is computed three times against the same allocated graph, so a
# fault that only shows on re-execution is caught -- both backend defects found
# in this package were of that kind (the first run matched exactly and later
# ones did not).
#
# Default: the subset of ops ggmlR actually uses in its layers, which takes
# seconds. Set GGMLR_TEST_ALL_OPS=1 to run every registered case.

bod_vulkan_available <- function() {
  tryCatch(ggml_vulkan_available() && ggml_vulkan_device_count() > 0,
           error = function(e) FALSE)
}

# Ops exercised by the layer builders. Kept short so the default run stays cheap;
# the full list is available behind GGMLR_TEST_ALL_OPS.
#
# The mul_mat variants are all in the default set on purpose. Each one selects a
# different kernel -- F16 weights, the tiled path that the vendor-specific
# branches live on (coopmat1/coopmat2/MMQ/subgroup shuffle), strided reads, and
# broadcast over a batch -- and a matmul that silently disagrees with the CPU on
# one of them is exactly the failure this file exists to catch. Gating them
# behind an environment variable would mean they are not run before a release.
bod_core_ops <- c("mul_mat", "mul_mat_batched", "mul_mat_f16", "mul_mat_large",
                  "mul_mat_unaligned", "mul_mat_noncont", "mul_mat_broadcast",
                  "mul_mat_id", "mul_mat_id_vec",
                  "mul_mat_vec_p021", "mul_mat_vec_nc",
                  "add", "mul_broadcast", "relu", "soft_max",
                  "permute_cont", "im2col_1d", "conv_2d",
                  "batch_norm_infer", "batch_norm_train",
                  # In the default set, not just under GGMLR_TEST_ALL_OPS: the
                  # Vulkan shader for this one disagreed with the CPU kernel by
                  # 1.55 on BoTNet26t, and nothing here noticed because the op
                  # was not covered at all. rel_pos_bias_rect is the load-bearing
                  # one -- H != W is what pins down which weight half goes with
                  # which axis, and a square case passes with them swapped.
                  "rel_pos_bias", "rel_pos_bias_rect", "rel_pos_bias_botnet")

# An op that genuinely diverges shows an NMSE far above this; f32 arithmetic
# reordering on a GPU stays well below it.
BOD_TOL <- 1e-6

test_that("backend ops: Vulkan agrees with CPU on the ops ggmlR builds", {
  skip_if_not(bod_vulkan_available(), "No Vulkan GPU available")

  be <- ggml_vulkan_init(0)
  on.exit(ggml_backend_free(be), add = TRUE)

  run_all <- identical(Sys.getenv("GGMLR_TEST_ALL_OPS"), "1")
  res <- ggml_test_backend_ops(be)

  expect_true(is.data.frame(res))
  expect_true(nrow(res) > 0)

  if (!run_all) {
    res <- res[res$op %in% bod_core_ops, , drop = FALSE]
    expect_true(nrow(res) > 0)
  }

  # An op the backend does not implement is reported, not failed: falling back to
  # the CPU for it is legitimate. Anything else that could not run is a failure --
  # it means the graph would not build or allocate.
  unsupported <- res[!is.na(res$note) & res$note == "unsupported by backend", ,
                     drop = FALSE]
  if (nrow(unsupported) > 0) {
    message("ops not implemented by this backend: ",
            paste(unsupported$op, collapse = ", "))
  }
  broken <- res[is.na(res$nmse) &
                  !(res$note %in% c("unsupported by backend")), , drop = FALSE]
  expect_equal(nrow(broken), 0L,
               info = paste("ops that failed to run:",
                            paste(sprintf("%s (%s)", broken$op, broken$note),
                                  collapse = ", ")))

  bad <- res[!is.na(res$nmse) & res$nmse > BOD_TOL, , drop = FALSE]
  expect_equal(nrow(bad), 0L,
               info = paste("ops disagreeing with CPU:",
                            paste(sprintf("%s nmse=%.3e", bad$op, bad$nmse),
                                  collapse = ", ")))
})

test_that("adamw step: the weight trajectory matches the CPU", {
  # Training with Adam on Vulkan was seen to drift from the CPU: step 1 agreed
  # bit-for-bit and the difference grew from step 2 onwards, ending in a
  # materially worse model (accuracy 0.53 vs 0.72 over ten epochs) while SGD
  # stayed exact. A single-execution comparison cannot see this, hence the
  # step-by-step trajectory.
  skip_if_not(bod_vulkan_available(), "No Vulkan GPU available")

  be <- ggml_vulkan_init(0)
  on.exit(ggml_backend_free(be), add = TRUE)

  res <- ggml_test_adamw_steps(be)
  skip_if(!is.null(res$note), paste("adamw step unavailable:", res$note[1]))

  expect_true(all(is.finite(res$w_cpu)))
  expect_true(all(is.finite(res$w_backend)))

  # The weights are a sum over 64 values of order 1, so a genuine disagreement
  # shows well above f32 summation noise.
  tol <- 1e-4
  bad <- res[res$abs_diff > tol, , drop = FALSE]
  expect_equal(nrow(bad), 0L,
               info = paste0("AdamW diverges from the CPU at step(s): ",
                             paste(sprintf("%d (|diff|=%.3e)", bad$step, bad$abs_diff),
                                   collapse = ", ")))
})

test_that("backend ops: the filter selects a subset", {
  skip_if_not(bod_vulkan_available(), "No Vulkan GPU available")

  be <- ggml_vulkan_init(0)
  on.exit(ggml_backend_free(be), add = TRUE)

  all_res <- ggml_test_backend_ops(be)
  one     <- ggml_test_backend_ops(be, filter = "conv_2d")

  expect_lt(nrow(one), nrow(all_res))
  expect_true(all(grepl("conv_2d", one$op)))
})

test_that("backend ops: CPU compared against itself is exact", {
  # Guards the harness rather than a backend: identical inputs through the same
  # implementation must produce identical outputs, so any non-zero error here
  # means the comparison itself is unsound (e.g. inputs not seeded alike).
  cpu <- ggml_backend_cpu_init()
  on.exit(ggml_backend_free(cpu), add = TRUE)

  res <- ggml_test_backend_ops(cpu)

  expect_true(nrow(res) > 0)
  broken <- res[is.na(res$nmse) & res$note != "unsupported by backend", ,
                drop = FALSE]
  expect_equal(nrow(broken), 0L,
               info = paste("cases that failed to run:",
                            paste(sprintf("%s (%s)", broken$op, broken$note),
                                  collapse = ", ")))
  ok <- res[!is.na(res$nmse), , drop = FALSE]
  expect_equal(max(ok$nmse), 0,
               info = paste("non-zero self-comparison for:",
                            paste(ok$op[ok$nmse > 0], collapse = ", ")))
})
