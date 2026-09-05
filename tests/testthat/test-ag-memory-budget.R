# ag_estimate_training_memory(): the four terms and how they scale.
#
# The estimate is only useful if its terms move the way the real thing does:
# optimizer state is a multiple of the weights, activations follow the batch,
# and everything is counted in double rather than f32. Each of those is a
# separate assertion below, because getting any one wrong makes the number
# confidently wrong rather than obviously broken.

test_that("weights and gradients are one 8-byte copy each", {
  r <- ag_estimate_training_memory(list(c(100, 100)), optimizer = "sgd_plain",
                                   activation_frac = 0, quiet = TRUE)
  expect_equal(r$params, 10000)
  expect_equal(r$weights, 10000 * 8)
  expect_equal(r$gradients, r$weights)
  expect_equal(r$optimizer, 0)
  expect_equal(r$total, r$weights + r$gradients)
})

test_that("the optimizer multiplier matches what each optimizer keeps", {
  shapes <- list(c(64, 64))
  adam  <- ag_estimate_training_memory(shapes, optimizer = "adam",      activation_frac = 0, quiet = TRUE)
  sgd   <- ag_estimate_training_memory(shapes, optimizer = "sgd",       activation_frac = 0, quiet = TRUE)
  plain <- ag_estimate_training_memory(shapes, optimizer = "sgd_plain", activation_frac = 0, quiet = TRUE)

  expect_equal(adam$optimizer,  2 * adam$weights)   # m and v
  expect_equal(sgd$optimizer,   1 * sgd$weights)    # velocity
  expect_equal(plain$optimizer, 0)                  # nothing
  expect_gt(adam$total, sgd$total)
  expect_gt(sgd$total, plain$total)
})

test_that("activations scale with the batch and nothing else does", {
  shapes <- list(c(128, 128))
  b1  <- ag_estimate_training_memory(shapes, batch_size = 1L,  quiet = TRUE)
  b64 <- ag_estimate_training_memory(shapes, batch_size = 64L, quiet = TRUE)

  expect_equal(b1$weights,   b64$weights)     # weights do not follow the batch
  expect_equal(b1$optimizer, b64$optimizer)   # nor does optimizer state
  expect_equal(b64$activations, 64 * b1$activations)
  expect_gt(b64$total, b1$total)
})

test_that("the count is in double, not f32", {
  # The whole point of the function: an f32 estimate would be half this. If
  # someone changes the constant to 4 believing ag_dtype() applies, this fails.
  r <- ag_estimate_training_memory(list(c(1000, 1000)), optimizer = "sgd_plain",
                                   activation_frac = 0, quiet = TRUE)
  expect_equal(r$bytes_per_scalar, 8)
  expect_equal(r$weights, 1e6 * 8)
  expect_match(r$note, "double")
})

test_that("shapes accumulate across parameters and accept a bare vector", {
  many <- ag_estimate_training_memory(list(c(10, 10), c(10, 10), c(10)),
                                      activation_frac = 0, quiet = TRUE)
  expect_equal(many$params, 210)

  one <- ag_estimate_training_memory(c(10, 10), activation_frac = 0, quiet = TRUE)
  expect_equal(one$params, 100)
})

test_that("bad input is refused rather than silently estimated", {
  expect_error(ag_estimate_training_memory(list()), "non-empty")
  expect_error(ag_estimate_training_memory(list(c(-1, 4))), "positive")
  expect_error(ag_estimate_training_memory(list(c(4, NA))), "positive")
  expect_error(ag_estimate_training_memory(list(c(4, 4)), optimizer = "rmsprop"))
})

test_that("the report is printed, and quiet suppresses it", {
  expect_output(ag_estimate_training_memory(list(c(8, 8))), "Training memory estimate")
  # The double-vs-f32 caveat is the most useful line in the report, so assert
  # it is actually printed rather than only reachable through $note.
  expect_output(ag_estimate_training_memory(list(c(8, 8))), "note:")

  # Every other test here runs quiet; a regression that started printing again
  # would only show up as noise in the log, which nobody fixes.
  expect_silent(r <- ag_estimate_training_memory(list(c(8, 8)), quiet = TRUE))
  expect_gt(r$total, 0)
  expect_match(r$note, "double")   # the caveat survives quiet, via the field
})

# --- the dtype fix in ggml_estimate_memory ----------------------------------
#
# Separate concern, same session: that function sizes one tensor in a ggml
# buffer, where dtype IS real. It used to return 4 bytes per element for every
# type, including quantised ones.

test_that("ggml_estimate_memory respects the tensor type", {
  n   <- 1000
  f32 <- ggml_estimate_memory(GGML_TYPE_F32, n, n)
  f16 <- ggml_estimate_memory(GGML_TYPE_F16, n, n)

  # F16 is half of F32, not equal to it as the old hardcoded switch had it.
  # Compare the data part: both carry the same fixed overhead + alignment.
  overhead <- f32 - n * n * 4
  expect_equal(f16 - overhead, n * n * 2)
})

test_that("ggml_estimate_memory accounts for quantised block sizes", {
  n    <- 1024
  f32  <- ggml_estimate_memory(GGML_TYPE_F32,  n, n)
  q4   <- ggml_estimate_memory(GGML_TYPE_Q4_0, n, n)
  # Q4_0 packs 32 values into an 18-byte block: well under a byte per element,
  # so a quantised tensor must come out far smaller than the F32 one.
  expect_lt(q4, f32 / 4)
})
