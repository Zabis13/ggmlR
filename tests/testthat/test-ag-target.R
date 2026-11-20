# Target networks (R/ag_target.R): ag_soft_update / ag_hard_update.

tg_lists <- function(seed = 41) {
  set.seed(seed)
  list(t = list(W = ag_param(matrix(rnorm(6), 2, 3)), b = ag_param(matrix(rnorm(2), 2, 1))),
       s = list(W = ag_param(matrix(rnorm(6), 2, 3)), b = ag_param(matrix(rnorm(2), 2, 1))))
}
vals <- function(ps) lapply(ps, as.matrix)

test_that("soft update is tau * source + (1 - tau) * target; target written, source not", {
  d <- tg_lists(); t0 <- vals(d$t); s0 <- vals(d$s)
  ag_soft_update(d$t, d$s, 0.25)
  for (nm in names(t0)) {
    expect_equal(as.matrix(d$t[[nm]]), 0.25 * s0[[nm]] + 0.75 * t0[[nm]], tolerance = 1e-12)
    expect_equal(as.matrix(d$s[[nm]]), s0[[nm]])
  }
})

test_that("tau = 0 changes nothing, tau = 1 and hard update copy exactly", {
  d <- tg_lists(); t0 <- vals(d$t)
  ag_soft_update(d$t, d$s, 0)
  expect_identical(vals(d$t), t0)
  ag_soft_update(d$t, d$s, 1)
  expect_identical(vals(d$t), vals(d$s))
  d <- tg_lists(2)
  ag_hard_update(d$t, d$s)
  expect_identical(vals(d$t), vals(d$s))
})

test_that("pairing: by name when named, by position otherwise", {
  d <- tg_lists(); s0 <- vals(d$s)
  ag_hard_update(d$t, rev(d$s))                    # named: order does not matter
  expect_identical(vals(d$t), s0)
  d <- tg_lists(3); s0 <- vals(d$s)
  ag_hard_update(unname(d$t), unname(d$s))         # positional
  expect_identical(unname(vals(d$t)), unname(s0))
})

test_that("bad input is refused before anything is written", {
  d <- tg_lists(); t0 <- vals(d$t)
  expect_error(ag_soft_update(d$t, d$s, 1.5), "\\[0, 1\\]")
  expect_error(ag_soft_update(d$t, d$s, -0.1), "\\[0, 1\\]")
  expect_error(ag_hard_update(d$t, d$s["W"]), "2 parameters, source has 1")
  bad <- d$s; names(bad) <- c("W", "c")
  expect_error(ag_hard_update(d$t, bad), "names differ")
  # second pair has the wrong shape: the first must not have been written
  bad <- list(W = d$s$W, b = ag_param(matrix(0, 3, 1)))
  expect_error(ag_hard_update(d$t, bad), "shape mismatch for parameter 'b'")
  expect_identical(vals(d$t), t0)
  expect_error(ag_hard_update(d$t, list(1, 2)), "ag_tensors only")
})

test_that("models: parameters Polyak-averaged, batch-norm buffers copied", {
  set.seed(5)
  mk <- function() ag_sequential(ag_linear(3L, 4L), ag_batch_norm(4L), ag_linear(4L, 2L))
  tgt <- mk(); src <- mk()
  src$layers[[2L]]$running_mean <- matrix(1:4 / 10, 4, 1)
  src$layers[[2L]]$running_var  <- matrix(4:1, 4, 1)
  tp0 <- vals(tgt$parameters()); sp0 <- vals(src$parameters())
  ag_soft_update(tgt, src, 0.1)
  tp <- vals(tgt$parameters())
  for (nm in names(tp0)) {
    expect_equal(tp[[nm]], 0.1 * sp0[[nm]] + 0.9 * tp0[[nm]], tolerance = 1e-12, info = nm)
    # strictly between the old target and the source where they differ
    dif <- tp0[[nm]] != sp0[[nm]]
    expect_true(all((tp[[nm]] - tp0[[nm]])[dif] * (sp0[[nm]] - tp[[nm]])[dif] > 0), info = nm)
  }
  expect_identical(tgt$layers[[2L]]$running_mean, src$layers[[2L]]$running_mean)
  expect_identical(tgt$layers[[2L]]$running_var,  src$layers[[2L]]$running_var)
})

# ---- GPU ----------------------------------------------------------------------

skip_if_no_gpu_tg <- function() {
  skip_if_not(ggml_vulkan_available() && ggml_vulkan_device_count() >= 1L,
              "no Vulkan device")
}

test_that("GPU: one launch, no host transfer, same values as the CPU", {
  skip_if_no_gpu_tg()
  ag_device("gpu")
  on.exit(ag_device("cpu"), add = TRUE)
  d <- tg_lists(); t0 <- vals(d$t); s0 <- vals(d$s)
  d <- tg_lists()                                  # fresh, on the device
  expect_false(is.null(ggmlR:::.ag_handle_of(d$t$W)))
  ggmlR:::ag_launch_count(reset = TRUE)
  ggmlR:::ag_xfer_count(TRUE)
  on.exit(ggmlR:::ag_xfer_count(FALSE), add = TRUE)
  ag_soft_update(d$t, d$s, 0.3)
  expect_equal(ggmlR:::ag_launch_count()[["total"]], 1)
  expect_length(ls(ggmlR:::.ag_xfer$counts), 0L)
  ggmlR:::ag_xfer_count(FALSE)
  for (nm in names(t0))
    expect_equal(as.matrix(d$t[[nm]]), 0.3 * s0[[nm]] + 0.7 * t0[[nm]],
                 tolerance = 1e-6, info = nm)
  ag_hard_update(d$t, d$s)
  for (nm in names(s0))
    expect_equal(as.matrix(d$t[[nm]]), as.matrix(d$s[[nm]]), info = nm)
})

test_that("GPU: soft updates repeated in a loop track the CPU", {
  skip_if_no_gpu_tg()
  run <- function(dev) {
    ag_device(dev)
    on.exit(ag_device("cpu"), add = TRUE)
    d <- tg_lists(9)
    for (k in 1:20) ag_soft_update(d$t, d$s, 0.05)
    vals(d$t)
  }
  cpu <- run("cpu"); gpu <- run("gpu")
  for (nm in names(cpu)) expect_equal(gpu[[nm]], cpu[[nm]], tolerance = 1e-5, info = nm)
})
