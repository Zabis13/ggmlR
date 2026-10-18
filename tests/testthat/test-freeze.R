# Freezing by name prefix, and fixing batch-norm statistics.

# ============================================================================
# prefix
# ============================================================================

test_that("prefix selects every layer whose name starts with it", {
  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_dense(8L, name = "enc_1") |>
    ggml_layer_dense(8L, name = "enc_2") |>
    ggml_layer_dense(2L, name = "head")
  m <- ggml_freeze_weights(ggml_model(inputs = x, outputs = out), prefix = "enc_")
  frozen_names <- vapply(Filter(function(nd) isFALSE(m$frozen_nodes[[nd$id]]),
                                ggmlR:::nn_topo_sort(m$outputs)),
                         function(nd) nd$config$name, character(1))
  expect_setequal(frozen_names, c("enc_1", "enc_2"))

  m <- ggml_unfreeze_weights(m, prefix = "enc_2")
  expect_null(m$frozen_nodes[[out$parents[[1L]]$id]])

  s <- ggml_model_sequential() |>
    ggml_layer_dense(8L, input_shape = 4L, name = "enc_1") |>
    ggml_layer_dense(2L, name = "head")
  s <- ggml_freeze_weights(s, prefix = "enc_")
  expect_false(s$layers[[1L]]$trainable)
  expect_true(s$layers[[2L]]$trainable)
})

test_that("a prefix stops at the separator: block1_ is not block10_", {
  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_dense(4L, name = "block1_ff") |>
    ggml_layer_dense(4L, name = "block10_ff") |>
    ggml_layer_dense(2L, name = "head")
  m <- ggml_freeze_weights(ggml_model(inputs = x, outputs = out), prefix = "block1_")
  b10 <- out$parents[[1L]]
  expect_false(m$frozen_nodes[[b10$parents[[1L]]$id]])   # block1_ff
  expect_null(m$frozen_nodes[[b10$id]])                   # block10_ff
})

test_that("layer_names and prefix together select the union", {
  s <- ggml_model_sequential() |>
    ggml_layer_dense(4L, input_shape = 4L, name = "enc_1") |>
    ggml_layer_dense(4L, name = "mid") |>
    ggml_layer_dense(2L, name = "head")
  s <- ggml_freeze_weights(s, layer_names = "head", prefix = "enc_")
  expect_equal(vapply(s$layers, `[[`, logical(1), "trainable"), c(FALSE, TRUE, FALSE))
})

test_that("a prefix or name that matches nothing is an error", {
  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_dense(2L, name = "head")
  m   <- ggml_model(inputs = x, outputs = out)
  expect_error(ggml_freeze_weights(m, prefix = "enc_"), "starts with 'enc_'")
  expect_error(ggml_freeze_weights(m, layer_names = "nope"), "nope")
})

test_that("prefix freezes a whole transformer block, and only it", {
  set.seed(1)
  x   <- ggml_input(shape = c(6L, 8L))
  h   <- ggml_layer_transformer_block(x, 8L, n_heads = 2L, ff_dim = 16L,
                                      name = "block1")
  h   <- ggml_layer_sequence_pooling(h, mode = "mean")
  out <- ggml_layer_dense(h, 2L, name = "head")
  m   <- ggml_compile(ggml_model(inputs = x, outputs = out),
                      optimizer = "adam", loss = "mse", backend = "cpu")
  xs <- array(rnorm(16L * 6L * 8L), c(16L, 6L, 8L))
  ys <- matrix(rnorm(32L), 16L, 2L)
  m  <- ggml_fit(m, xs, ys, epochs = 1L, batch_size = 16L, verbose = 0L)
  m  <- ggml_freeze_weights(m, prefix = "block1_")

  f <- tempfile(fileext = ".rds"); on.exit(unlink(f))
  ggml_save_model(m, f)
  before <- readRDS(f)$node_weights_data
  m <- ggml_fit(m, xs, ys, epochs = 3L, batch_size = 16L, verbose = 0L)
  ggml_save_model(m, f)
  after <- readRDS(f)$node_weights_data

  ids <- function(prefix) Filter(function(nd) {
    nm <- nd$config$name
    !is.null(nm) && startsWith(nm, prefix)
  }, ggmlR:::nn_topo_sort(m$outputs))
  block_ids <- vapply(ids("block1_"), `[[`, character(1), "id")
  block_ids <- intersect(block_ids, names(before))   # the ones with weights
  expect_gte(length(block_ids), 5L)                  # norm1, attn, norm2, ff1, ff2
  for (id in block_ids) expect_equal(after[[id]], before[[id]])
  expect_false(isTRUE(all.equal(after[[out$id]], before[[out$id]])))
})

# ============================================================================
# batch-norm statistics
# ============================================================================

bn_model <- function() {
  set.seed(2)
  m <- ggml_model_sequential() |>
    ggml_layer_dense(8L, input_shape = 4L) |>
    ggml_layer_batch_norm() |>
    ggml_layer_dense(2L)
  ggml_compile(m, optimizer = "adam", loss = "mse", backend = "cpu")
}
bn_data <- function(n = 32L) {
  set.seed(3)
  list(x = matrix(rnorm(n * 4L, mean = 2), n, 4L), y = matrix(rnorm(n * 2L), n, 2L))
}
bn_get <- function(m, what) ggml_backend_tensor_get_data(m$layers[[2L]]$weights[[what]])

test_that("fixed statistics are not recalibrated, while gamma still trains", {
  d <- bn_data()
  m <- ggml_fit(bn_model(), d$x, d$y, epochs = 2L, batch_size = 32L, verbose = 0L)
  m <- ggml_freeze_bn_stats(m)
  rm0 <- bn_get(m, "running_mean"); rv0 <- bn_get(m, "running_var")
  g0  <- bn_get(m, "gamma")

  m <- ggml_fit(m, d$x * 3, d$y, epochs = 3L, batch_size = 32L, verbose = 0L)
  expect_equal(bn_get(m, "running_mean"), rm0)
  expect_equal(bn_get(m, "running_var"),  rv0)
  expect_false(isTRUE(all.equal(bn_get(m, "gamma"), g0)))

  m <- ggml_unfreeze_bn_stats(m)
  m <- ggml_fit(m, d$x * 3, d$y, epochs = 1L, batch_size = 32L, verbose = 0L)
  expect_false(isTRUE(all.equal(bn_get(m, "running_mean"), rm0)))
})

test_that("with fixed statistics a row's training loss ignores the rest of the batch", {
  d <- bn_data()
  m <- ggml_fit(bn_model(), d$x, d$y, epochs = 2L, batch_size = 32L, verbose = 0L)
  x2 <- d$x; x2[-1L, ] <- x2[-1L, ] * 5 + 3          # row 1 alone stays the same
  w  <- c(1, rep(0, 31L))                            # loss of row 1 only

  row1_loss <- function(model) {
    # The unfixed model warns that a trainer does not update its statistics;
    # that is expected here and tested on its own below.
    tr <- suppressWarnings(ggml_trainer(model, batch_size = 32L, loss_mask = 1L))
    on.exit(tr$free())
    c(tr$eval(d$x, d$y, w), tr$eval(x2, d$y, w))
  }
  batch_stats <- row1_loss(m)
  fixed_stats <- row1_loss(ggml_freeze_bn_stats(m))
  expect_false(isTRUE(all.equal(batch_stats[1L], batch_stats[2L])))
  expect_equal(fixed_stats[1L], fixed_stats[2L])
})

test_that("fixed statistics survive save and load", {
  d <- bn_data()
  m <- ggml_fit(bn_model(), d$x, d$y, epochs = 1L, batch_size = 32L, verbose = 0L)
  m <- ggml_freeze_bn_stats(m)
  f <- tempfile(fileext = ".rds"); on.exit(unlink(f))
  ggml_save_model(m, f)
  expect_true(isTRUE(ggml_load_model(f, backend = "cpu")$layers[[2L]]$config$fixed_stats))
})

test_that("bn-stats calls name what they cannot do", {
  s <- ggml_model_sequential() |> ggml_layer_dense(2L, input_shape = 4L)
  expect_error(ggml_freeze_bn_stats(s), "no batch_norm layer")

  x   <- ggml_input(shape = 4L)
  out <- x |> ggml_layer_batch_norm() |> ggml_layer_dense(2L)
  expect_error(ggml_freeze_bn_stats(ggml_model(inputs = x, outputs = out)),
               "per-sample RMS")
  expect_error(ggml_freeze_bn_stats(list()), "got an object of class 'list'")
})

test_that("a trainer warns about unfixed batch-norm and $calibrate_bn sets the statistics", {
  d <- bn_data()
  expect_warning(tr <- ggml_trainer(bn_model(), batch_size = 32L),
                 "calibrate_bn")
  on.exit(tr$free())
  for (i in 1:3) tr$step(d$x, d$y)

  before <- bn_get(tr$model(), "running_mean")
  tr$calibrate_bn(d$x)
  after <- bn_get(tr$model(), "running_mean")
  expect_false(isTRUE(all.equal(before, after)))

  # The statistics of the first layer's output over d$x, as ggml_fit() computes.
  h <- ggml_predict(tr$model(), d$x, batch_size = 32L)   # runs; finite
  expect_true(all(is.finite(h)))

  m_fixed <- ggml_freeze_bn_stats(ggml_fit(bn_model(), d$x, d$y, epochs = 1L,
                                           batch_size = 32L, verbose = 0L))
  expect_no_warning(tr2 <- ggml_trainer(m_fixed, batch_size = 32L))
  tr2$free()
})
