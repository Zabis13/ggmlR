# Tests for multi-output training (several loss heads, weighted sum)

cleanup_mo_model <- function(model) {
  if (!is.null(model$compilation$buffer)) {
    ggml_backend_buffer_free(model$compilation$buffer)
  }
  if (!is.null(model$compilation$ctx_weights)) {
    ggml_free(model$compilation$ctx_weights)
  }
  if (!is.null(model$compilation$sched)) {
    ggml_backend_sched_free(model$compilation$sched)
  }
  if (!is.null(model$compilation$backend)) {
    ggml_backend_free(model$compilation$backend)
  }
  if (!is.null(model$compilation$cpu_backend)) {
    ggml_backend_free(model$compilation$cpu_backend)
  }
}

# Two-head model: a shared trunk feeding a regression head and a second head.
build_two_head <- function(units_a = 2L, units_b = 1L,
                           act_a = NULL, act_b = NULL) {
  inp   <- ggml_input(shape = 4L, name = "in")
  trunk <- ggml_apply(inp, ggml_dense(8L, activation = "relu", name = "trunk"))
  outa  <- ggml_apply(trunk, ggml_dense(units_a, activation = act_a, name = "head_a"))
  outb  <- ggml_apply(trunk, ggml_dense(units_b, activation = act_b, name = "head_b"))
  ggml_model(inputs = inp, outputs = list(outa, outb))
}

make_xy <- function(n = 64L) {
  set.seed(42L)
  x  <- matrix(runif(n * 4L), nrow = n)
  ya <- cbind(rowSums(x[, 1:2]), rowSums(x[, 3:4]))
  yb <- matrix(rowSums(x), ncol = 1L)
  list(x = x, ya = ya, yb = yb)
}

# nn_resolve_losses() is internal, so reach it through the namespace: the file
# then runs the same under test_dir() and a bare test_file().
nn_resolve_losses <- ggmlR:::nn_resolve_losses

test_that("loss resolution matches outputs by name, not position", {
  spec <- nn_resolve_losses(
    list(head_b = "mse", head_a = "mse"), NULL, c("head_a", "head_b"))
  expect_equal(vapply(spec, function(s) s$name, character(1)),
               c("head_a", "head_b"))

  spec_w <- nn_resolve_losses("mse", c(head_b = 0.5, head_a = 2),
                              c("head_a", "head_b"))
  expect_equal(vapply(spec_w, function(s) s$weight, numeric(1)), c(2, 0.5))
})

test_that("a single loss name is broadcast to every head", {
  spec <- nn_resolve_losses("mse", NULL, c("a", "b", "c"))
  expect_length(spec, 3L)
  expect_true(all(vapply(spec, function(s) s$loss, character(1)) == "mse"))
  expect_true(all(vapply(spec, function(s) s$weight, numeric(1)) == 1))
})

test_that("mismatched loss / loss_weights lengths are rejected", {
  expect_error(nn_resolve_losses(c("mse", "mse"), NULL, c("a", "b", "c")),
               "one entry per output")
  expect_error(nn_resolve_losses("mse", c(1, 2, 3), c("a", "b")),
               "one entry per output")
  expect_error(nn_resolve_losses(list(nope = "mse"), NULL, c("a")),
               "do not match model outputs")
})

test_that("sequential models reject multi-output compile arguments", {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(2L, activation = "softmax", input_shape = 4L)
  expect_error(ggml_compile(m, loss = "mse", loss_weights = c(1, 1)),
               "loss_weights")
  expect_error(ggml_compile(m, loss = list("mse", "mse")),
               "single output")
})

test_that("BOTH heads train: each head's loss falls", {
  skip_on_cran()
  d <- make_xy()
  # seed before compile: weights are drawn at compile time
  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, optimizer = "adam", loss = "mse",
                        backend = "cpu")
  model <- ggml_fit(model, d$x, list(head_a = d$ya, head_b = d$yb),
                    epochs = 12L, batch_size = 16L, verbose = 0L)

  h <- model$history
  expect_true(!is.null(h$train_head_a_loss))
  expect_true(!is.null(h$train_head_b_loss))

  # This is the regression guard for "only the last output is trained":
  # before multi-loss, head_a would never have improved.
  expect_lt(tail(h$train_head_a_loss, 1L), h$train_head_a_loss[1L])
  expect_lt(tail(h$train_head_b_loss, 1L), h$train_head_b_loss[1L])

  cleanup_mo_model(model)
})

test_that("a zero weight freezes its head while the other still trains", {
  skip_on_cran()
  d <- make_xy()
  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, optimizer = "adam", loss = "mse",
                        loss_weights = c(head_a = 1, head_b = 0),
                        backend = "cpu")
  model <- ggml_fit(model, d$x, list(head_a = d$ya, head_b = d$yb),
                    epochs = 10L, batch_size = 16L, verbose = 0L)

  h <- model$history
  expect_lt(tail(h$train_head_a_loss, 1L), h$train_head_a_loss[1L])
  # head_b gets no gradient of its own. Its loss can still move, because the
  # shared trunk keeps changing underneath it, so this only asserts that the
  # weighted total ignores it: the total must equal head_a's loss.
  expect_equal(tail(h$train_loss, 1L), tail(h$train_head_a_loss, 1L),
               tolerance = 1e-4)

  cleanup_mo_model(model)
})

test_that("y as a list must have one entry per output", {
  skip_on_cran()
  d <- make_xy()
  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, loss = "mse", backend = "cpu")

  expect_error(ggml_fit(model, d$x, list(d$ya), epochs = 1L, batch_size = 16L,
                        verbose = 0L),
               "one entry per output")

  cleanup_mo_model(model)
})

test_that("a plain matrix y trains the last output only (legacy behaviour)", {
  skip_on_cran()
  d <- make_xy()
  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, loss = "mse", backend = "cpu")

  # head_b is the last output and has 1 column, so a 1-column y is accepted
  # and trains it; the other output stays an exposed activation.
  model <- ggml_fit(model, d$x, d$yb, epochs = 5L, batch_size = 16L,
                    verbose = 0L)
  expect_true(all(is.finite(model$history$train_loss)))
  expect_lt(tail(model$history$train_loss, 1L), model$history$train_loss[1L])
  # Single trained head -> no per-head columns.
  expect_null(model$history$train_head_a_loss)

  cleanup_mo_model(model)
})

test_that("a plain matrix y of the wrong width is rejected, not aborted", {
  skip_on_cran()
  d <- make_xy()
  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, loss = "mse", backend = "cpu")

  # d$ya has 2 columns but the trained (last) output has 1. Before this check
  # the mismatch reached ggml and aborted the R process.
  expect_error(ggml_fit(model, d$x, d$ya, epochs = 1L, batch_size = 16L,
                        verbose = 0L),
               "trained output")

  cleanup_mo_model(model)
})

test_that("evaluate reports per-head losses and the weighted total", {
  skip_on_cran()
  d <- make_xy()
  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, loss = "mse",
                        loss_weights = c(head_a = 1, head_b = 0.5),
                        backend = "cpu")
  model <- ggml_fit(model, d$x, list(head_a = d$ya, head_b = d$yb),
                    epochs = 3L, batch_size = 16L, verbose = 0L)

  res <- ggml_evaluate(model, d$x, list(head_a = d$ya, head_b = d$yb))
  expect_true(!is.null(res$head_a_loss))
  expect_true(!is.null(res$head_b_loss))
  expect_equal(res$loss, res$head_a_loss + 0.5 * res$head_b_loss,
               tolerance = 1e-5)

  cleanup_mo_model(model)
})

test_that("a CE head and an MSE head can be mixed", {
  skip_on_cran()
  d <- make_xy()
  # head_a is a 2-class softmax head, head_b stays a plain regression head.
  ya_onehot <- cbind(as.numeric(d$ya[, 1] > 1), as.numeric(d$ya[, 1] <= 1))

  set.seed(1L)
  model <- build_two_head(units_a = 2L, act_a = "softmax")
  model <- ggml_compile(model,
                        loss = list(head_a = "categorical_crossentropy",
                                    head_b = "mse"),
                        backend = "cpu")
  model <- ggml_fit(model, d$x, list(head_a = ya_onehot, head_b = d$yb),
                    epochs = 10L, batch_size = 16L, verbose = 0L)

  h <- model$history
  # Only the CE head's softmax is stripped for training; the MSE head keeps
  # its (absent) activation, so both are optimized on the right quantity.
  expect_lt(tail(h$train_head_a_loss, 1L), h$train_head_a_loss[1L])
  expect_lt(tail(h$train_head_b_loss, 1L), h$train_head_b_loss[1L])

  cleanup_mo_model(model)
})

test_that("single-output models are unaffected", {
  skip_on_cran()
  d <- make_xy()
  set.seed(1L)
  inp <- ggml_input(shape = 4L)
  out <- ggml_apply(inp, ggml_dense(1L, name = "only"))
  model <- ggml_model(inputs = inp, outputs = out)
  model <- ggml_compile(model, loss = "mse", backend = "cpu")
  model <- ggml_fit(model, d$x, d$yb, epochs = 5L, batch_size = 16L,
                    verbose = 0L)

  h <- model$history
  expect_lt(tail(h$train_loss, 1L), h$train_loss[1L])
  # No per-head columns for a single-output model.
  expect_null(h$train_only_loss)

  cleanup_mo_model(model)
})

test_that("loss_weights survive a save/load round trip", {
  skip_on_cran()
  d <- make_xy()
  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, loss = "mse",
                        loss_weights = c(head_a = 1, head_b = 0.25),
                        backend = "cpu")
  model <- ggml_fit(model, d$x, list(head_a = d$ya, head_b = d$yb),
                    epochs = 2L, batch_size = 16L, verbose = 0L)

  path <- tempfile(fileext = ".rds")
  ggml_save_model(model, path)
  cleanup_mo_model(model)

  loaded <- ggml_load_model(path, backend = "cpu")
  w <- vapply(loaded$compilation$loss_spec, function(s) s$weight, numeric(1))
  expect_equal(unname(w), c(1, 0.25))

  cleanup_mo_model(loaded)
  unlink(path)
})

# ---- Vulkan (GPU-first) ------------------------------------------------------
#
# The reduction over heads (scale + add) and the per-head loss subgraphs run on
# whatever backend the scheduler picks, so the GPU path needs its own coverage:
# a mixed-backend split that dropped a head's gradient would still "train" and
# only show up as a wrong number.

skip_if_no_vulkan <- function() {
  skip_if(!ggml_vulkan_available(), "Vulkan not available")
  skip_if(ggml_vulkan_device_count() == 0L, "No Vulkan devices")
}

test_that("multi-output training on Vulkan matches the CPU", {
  skip_on_cran()
  skip_if_no_vulkan()
  d <- make_xy()

  fit_on <- function(be) {
    set.seed(1L)
    m <- build_two_head()
    m <- ggml_compile(m, optimizer = "adam", loss = "mse", backend = be)
    m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                  epochs = 12L, batch_size = 16L, verbose = 0L)
    h <- m$history
    on.exit(cleanup_mo_model(m))
    c(a = tail(h$train_head_a_loss, 1L), b = tail(h$train_head_b_loss, 1L),
      total = tail(h$train_loss, 1L))
  }

  cpu <- fit_on("cpu")
  gpu <- fit_on("vulkan")

  # f32 accumulation differs in ordering between backends, so this is a
  # tolerance check, not bit-equality.
  expect_equal(unname(gpu["a"]),     unname(cpu["a"]),     tolerance = 1e-3)
  expect_equal(unname(gpu["b"]),     unname(cpu["b"]),     tolerance = 1e-3)
  expect_equal(unname(gpu["total"]), unname(cpu["total"]), tolerance = 1e-3)
})

test_that("the weighted reduction is correct on Vulkan", {
  skip_on_cran()
  skip_if_no_vulkan()
  d <- make_xy()

  set.seed(1L)
  m <- build_two_head()
  m <- ggml_compile(m, loss = "mse",
                    loss_weights = c(head_a = 1, head_b = 0.5),
                    backend = "vulkan")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 6L, batch_size = 16L, verbose = 0L)
  h <- m$history

  # The scalar the optimizer minimizes must equal sum(w_i * loss_i) as computed
  # from the per-head values read back off the GPU.
  expect_equal(tail(h$train_loss, 1L),
               tail(h$train_head_a_loss, 1L) + 0.5 * tail(h$train_head_b_loss, 1L),
               tolerance = 1e-5)

  cleanup_mo_model(m)
})

test_that("a CE head and an MSE head can be mixed on Vulkan", {
  skip_on_cran()
  skip_if_no_vulkan()
  d <- make_xy()
  ya_onehot <- cbind(as.numeric(d$ya[, 1] > 1), as.numeric(d$ya[, 1] <= 1))

  set.seed(1L)
  m <- build_two_head(units_a = 2L, act_a = "softmax")
  m <- ggml_compile(m, loss = list(head_a = "categorical_crossentropy",
                                   head_b = "mse"),
                    backend = "vulkan")
  m <- ggml_fit(m, d$x, list(head_a = ya_onehot, head_b = d$yb),
                epochs = 10L, batch_size = 16L, verbose = 0L)
  h <- m$history

  expect_true(all(is.finite(h$train_head_a_loss)))
  expect_true(all(is.finite(h$train_head_b_loss)))
  expect_lt(tail(h$train_head_b_loss, 1L), h$train_head_b_loss[1L])

  cleanup_mo_model(m)
})

test_that("four heads fit in the graph and the context on Vulkan", {
  skip_on_cran()
  skip_if_no_vulkan()
  # Guards the per-head sizing of ctx_static in ggml_opt_build(): it used to be
  # hardcoded to one loss, and overflowing it aborts the R process rather than
  # raising an error.
  set.seed(42L)
  n <- 64L
  x  <- matrix(runif(n * 4L), nrow = n)
  ys <- lapply(1:4, function(i) matrix(rowSums(x) * i, ncol = 1L))
  names(ys) <- paste0("h", 1:4)

  set.seed(1L)
  inp  <- ggml_input(shape = 4L, name = "in")
  tr   <- ggml_apply(inp, ggml_dense(8L, activation = "relu", name = "trunk"))
  outs <- lapply(1:4, function(i) ggml_apply(tr, ggml_dense(1L, name = paste0("h", i))))
  m <- ggml_model(inputs = inp, outputs = outs)
  m <- ggml_compile(m, loss = "mse", backend = "vulkan")
  m <- ggml_fit(m, x, ys, epochs = 5L, batch_size = 16L, verbose = 0L)

  h <- m$history
  per_head <- vapply(1:4, function(i) tail(h[[paste0("train_h", i, "_loss")]], 1L),
                     numeric(1))
  expect_length(per_head, 4L)
  expect_true(all(is.finite(per_head)))
  # All weights are 1, so the optimized total is the plain sum.
  expect_equal(tail(h$train_loss, 1L), sum(per_head), tolerance = 1e-4)

  cleanup_mo_model(m)
})

test_that("per-head accuracy is reported for CE heads and absent for others", {
  skip_on_cran()
  d <- make_xy()
  ya_onehot <- cbind(as.numeric(d$ya[, 1] > 1), as.numeric(d$ya[, 1] <= 1))

  set.seed(1L)
  m <- build_two_head(units_a = 2L, act_a = "softmax")
  m <- ggml_compile(m, loss = list(head_a = "categorical_crossentropy",
                                   head_b = "mse"),
                    backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = ya_onehot, head_b = d$yb),
                epochs = 5L, batch_size = 16L, verbose = 0L)
  h <- m$history

  # head_a is cross-entropy -> it has an accuracy, in [0, 1].
  expect_false(is.null(h$train_head_a_accuracy))
  expect_true(all(h$train_head_a_accuracy >= 0 & h$train_head_a_accuracy <= 1))
  # head_b is MSE -> accuracy is meaningless, so the column is not reported
  # (rather than being an all-NA column pretending to be a metric).
  expect_null(h$train_head_b_accuracy)

  cleanup_mo_model(m)
})

# ---------------------------------------------------------------------------
# Callbacks on the multi-output path
#
# Until the epoch loop moved from C into R, ggml_fit() on a multi-output model
# went straight through ggml_opt_fit_multi() and never called a callback at
# all: no early stopping, no LR schedule, no monitor=. These cover that the
# hook exists and that the per-head keys a callback monitors are the ones the
# history reports.
# ---------------------------------------------------------------------------

test_that("callbacks fire on the multi-output path", {
  skip_on_cran()
  d <- make_xy()
  seen <- list(begin = integer(0), end = integer(0))
  spy <- list(
    on_epoch_begin = function(epoch, logs, state) {
      seen$begin <<- c(seen$begin, epoch); invisible(NULL)
    },
    on_epoch_end = function(epoch, logs, state) {
      seen$end <<- c(seen$end, epoch); invisible(NULL)
    }
  )

  set.seed(1L)
  m <- build_two_head()
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 4L, batch_size = 16L, verbose = 0L,
                callbacks = list(spy))

  expect_equal(seen$begin, 1:4)
  expect_equal(seen$end,   1:4)

  cleanup_mo_model(m)
})

test_that("early stopping cuts multi-output training short", {
  skip_on_cran()
  d <- make_xy()
  # Stops at the end of epoch 2 regardless of the metric, so the assertion is
  # about the mechanism rather than about the loss happening to plateau.
  stopper <- list(
    on_epoch_end = function(epoch, logs, state) {
      if (epoch >= 2L) state$stop <- TRUE
      invisible(NULL)
    }
  )

  set.seed(1L)
  m <- build_two_head()
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 10L, batch_size = 16L, verbose = 0L,
                callbacks = list(stopper))
  h <- m$history

  # History is truncated to the epochs actually run -- aggregate and per-head
  # alike, so the columns stay the same length.
  expect_length(h$train_loss, 2L)
  expect_length(h$train_head_a_loss, 2L)
  expect_length(h$train_head_b_loss, 2L)
  expect_equal(h$epochs, 1:2)

  cleanup_mo_model(m)
})

test_that("a callback sees per-head keys, matching the history columns", {
  skip_on_cran()
  d <- make_xy()
  captured <- NULL
  grabber <- list(
    on_epoch_end = function(epoch, logs, state) {
      if (epoch == 1L) captured <<- logs
      invisible(NULL)
    }
  )

  set.seed(1L)
  m <- build_two_head()
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 2L, batch_size = 16L, verbose = 0L,
                validation_split = 0.25, callbacks = list(grabber))
  h <- m$history

  # Both phases are present, per head, under the prefixed names.
  expect_true(all(c("train_head_a_loss", "train_head_b_loss",
                    "val_head_a_loss", "val_head_b_loss") %in% names(captured)))
  expect_true(is.finite(captured$train_head_a_loss))
  expect_true(is.finite(captured$val_head_a_loss))

  # The key a monitor= would name is the same one the history reports -- a
  # mismatch here is silent, since a callback skips a NULL metric.
  for (key in c("train_head_a_loss", "train_head_b_loss",
                "val_head_a_loss", "val_head_b_loss")) {
    expect_false(is.null(h[[key]]), info = key)
    expect_equal(h[[key]][1L], captured[[key]], info = key)
  }

  cleanup_mo_model(m)
})

test_that("early stopping can monitor one head rather than the total", {
  skip_on_cran()
  d <- make_xy()

  # ggml_callback_early_stopping() skips a metric it cannot find -- silently,
  # by design -- so "did training stop early" cannot tell a resolved monitor
  # from an unknown one. Watch what the monitored key actually holds instead,
  # then check the callback acts on it.
  monitored <- numeric(0)
  watcher <- list(
    on_epoch_end = function(epoch, logs, state) {
      monitored <<- c(monitored, logs[["val_head_a_loss"]])
      invisible(NULL)
    }
  )
  # Stops as soon as the monitored head fails to improve; whether that happens
  # depends on the data, so it is not asserted here.
  stopper <- ggml_callback_early_stopping(monitor = "val_head_a_loss",
                                          patience = 1L)

  set.seed(1L)
  m <- build_two_head()
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 6L, batch_size = 16L, verbose = 0L,
                validation_split = 0.25,
                callbacks = list(watcher, stopper))
  h <- m$history

  # The monitored key resolved every epoch that ran -- an unknown key would
  # have left `monitored` full of NULLs, i.e. length 0 after c().
  expect_length(monitored, length(h$train_loss))
  expect_true(all(is.finite(monitored)))
  # ...and it is the same series the history reports under that name.
  expect_equal(monitored, h$val_head_a_loss)

  cleanup_mo_model(m)
})

test_that("a head named val_x does not collide with another head's val key", {
  skip_on_cran()
  d <- make_xy()

  # "val_a" is a legal output name, and without the train_ prefix on the
  # training keys it would produce "val_a_loss" -- the same key head "a" uses
  # for its validation loss, silently overwriting it.
  inp   <- ggml_input(shape = 4L, name = "in")
  trunk <- ggml_apply(inp, ggml_dense(8L, activation = "relu", name = "trunk"))
  out_a <- ggml_apply(trunk, ggml_dense(2L, name = "a"))
  out_v <- ggml_apply(trunk, ggml_dense(1L, name = "val_a"))

  set.seed(1L)
  m <- ggml_model(inputs = inp, outputs = list(out_a, out_v))
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d$x, list(a = d$ya, val_a = d$yb),
                epochs = 3L, batch_size = 16L, verbose = 0L,
                validation_split = 0.25)
  h <- m$history

  # Four distinct keys, none shadowing another.
  keys <- c("train_a_loss", "val_a_loss", "train_val_a_loss", "val_val_a_loss")
  for (key in keys) expect_false(is.null(h[[key]]), info = key)
  expect_equal(anyDuplicated(names(h)), 0L)

  # head "a" is 2-wide and head "val_a" 1-wide on different targets, so their
  # losses differ -- if the keys had collided, two of these would be identical.
  expect_false(isTRUE(all.equal(h$train_a_loss, h$train_val_a_loss)))

  cleanup_mo_model(m)
})

# ---------------------------------------------------------------------------
# Thread-count re-sync inside the R epoch loop
#
# A CPU backend is given ggml_set_n_threads() when it is created, and the R
# wrapper around sched_graph_compute re-applies it per compute -- but training
# never goes through that wrapper: ggml_opt_eval calls
# ggml_backend_sched_graph_compute straight from C. The single C entry points
# (ggml_opt_fit, ggml_opt_fit_multi) sync once before their loop; an R-side
# loop has no such moment, so it re-syncs per epoch itself. There is no way to
# read a backend's thread count back from R, so this covers the behaviour:
# changing the setting mid-training must be harmless.
# ---------------------------------------------------------------------------

test_that("changing thread count mid-training is harmless (multi-output)", {
  skip_on_cran()
  d <- make_xy()
  original <- ggml_get_n_threads()
  on.exit(ggml_set_n_threads(original), add = TRUE)

  flip <- list(
    on_epoch_begin = function(epoch, logs, state) {
      # Alternate between 1 and 2 threads so the per-epoch re-sync actually has
      # something to pick up.
      ggml_set_n_threads(if (epoch %% 2L == 0L) 2L else 1L)
      invisible(NULL)
    }
  )

  ggml_set_n_threads(1L)
  set.seed(1L)
  m <- build_two_head()
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 4L, batch_size = 16L, verbose = 0L,
                callbacks = list(flip))
  h <- m$history

  # Training still runs to completion and both heads still learn -- the thread
  # count affects how the work is split, never the arithmetic.
  expect_length(h$train_loss, 4L)
  expect_true(all(is.finite(h$train_loss)))
  expect_lt(tail(h$train_head_a_loss, 1L), h$train_head_a_loss[1L])
  expect_lt(tail(h$train_head_b_loss, 1L), h$train_head_b_loss[1L])

  cleanup_mo_model(m)
})

test_that("changing thread count mid-training is harmless (single-output)", {
  skip_on_cran()
  # The per-epoch re-sync landed in every R-side loop, including the
  # single-output path that was working before -- so it gets the same check.
  d <- make_xy()
  original <- ggml_get_n_threads()
  on.exit(ggml_set_n_threads(original), add = TRUE)

  flip <- list(
    on_epoch_begin = function(epoch, logs, state) {
      ggml_set_n_threads(if (epoch %% 2L == 0L) 2L else 1L)
      invisible(NULL)
    }
  )

  ggml_set_n_threads(1L)
  set.seed(1L)
  inp <- ggml_input(shape = 4L, name = "in")
  out <- ggml_apply(inp, ggml_dense(1L, name = "only"))
  m <- ggml_model(inputs = inp, outputs = out)
  m <- ggml_compile(m, loss = "mse", backend = "cpu")
  m <- ggml_fit(m, d$x, d$yb, epochs = 4L, batch_size = 16L, verbose = 0L,
                callbacks = list(flip))
  h <- m$history

  expect_length(h$train_loss, 4L)
  expect_true(all(is.finite(h$train_loss)))
  expect_lt(tail(h$train_loss, 1L), h$train_loss[1L])

  cleanup_mo_model(m)
})

# ---------------------------------------------------------------------------
# Multi-output on the multi-input path
#
# Two inputs mean there is no ggml_opt_dataset to batch from, so ggml_fit()
# drives its own batch loop and fills each head's labels tensor itself. These
# check that every head is actually trained there, not just the last one.
# ---------------------------------------------------------------------------

# Two inputs merged into a shared trunk, then two heads off that trunk.
build_two_in_two_head <- function(units_a = 2L, units_b = 1L,
                                  units = 8L,
                                  act_a = NULL, act_b = NULL) {
  in1   <- ggml_input(shape = 4L, name = "in1")
  in2   <- ggml_input(shape = 3L, name = "in2")
  b1    <- ggml_apply(in1, ggml_dense(units, activation = "relu", name = "b1"))
  b2    <- ggml_apply(in2, ggml_dense(units, activation = "relu", name = "b2"))
  trunk <- ggml_layer_add(list(b1, b2), name = "trunk")
  outa  <- ggml_apply(trunk, ggml_dense(units_a, activation = act_a, name = "head_a"))
  outb  <- ggml_apply(trunk, ggml_dense(units_b, activation = act_b, name = "head_b"))
  ggml_model(inputs = list(in1, in2), outputs = list(outa, outb))
}

make_xy_multi_in <- function(n = 64L) {
  set.seed(42L)
  x1 <- matrix(runif(n * 4L), nrow = n)
  x2 <- matrix(runif(n * 3L), nrow = n)
  # Each head depends on both inputs, so neither branch can be ignored.
  ya <- cbind(rowSums(x1[, 1:2]) + x2[, 1], rowSums(x1[, 3:4]) + x2[, 2])
  yb <- matrix(rowSums(x1) + rowSums(x2), ncol = 1L)
  list(x = list(x1, x2), ya = ya, yb = yb)
}

test_that("multi-output trains on the multi-input path", {
  d <- make_xy_multi_in()
  set.seed(1L)
  m <- build_two_in_two_head()
  m <- ggml_compile(m, loss = c("mse", "mse"), backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 6L, batch_size = 16L, verbose = 0L)
  h <- m$history

  expect_length(h$train_loss, 6L)
  expect_true(all(is.finite(h$train_loss)))
  # Both heads must be reported, and both must actually be learning -- the
  # aggregate loss can fall while a head that is not wired up stalls.
  expect_true(all(is.finite(h$train_head_a_loss)))
  expect_true(all(is.finite(h$train_head_b_loss)))
  expect_lt(tail(h$train_head_a_loss, 1L), h$train_head_a_loss[1L])
  expect_lt(tail(h$train_head_b_loss, 1L), h$train_head_b_loss[1L])

  cleanup_mo_model(m)
})

test_that("each head on the multi-input path gets its own label slice", {
  # Heads of different widths: head_a takes columns 1:2 of the concatenated
  # labels, head_b column 3. An off-by-one in the offsets trains head_b against
  # head_a's second column instead -- which still converges, just to the wrong
  # target -- so the check is which target each head tracks, not how far it got.
  d <- make_xy_multi_in(n = 256L)
  set.seed(1L)
  m <- build_two_in_two_head(units = 16L)
  m <- ggml_compile(m, loss = c("mse", "mse"), backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 200L, batch_size = 32L, verbose = 0L)

  preds <- ggml_predict(m, d$x, batch_size = 32L)
  expect_length(preds, 2L)
  expect_equal(dim(preds[[1L]]), c(nrow(d$ya), 2L))
  expect_equal(dim(preds[[2L]]), c(nrow(d$yb), 1L))

  pb <- as.vector(preds[[2L]])
  # head_b's target is neither column of head_a's, so it must track its own
  # more closely than either of head_a's -- the signature of a correct slice.
  expect_gt(cor(pb, as.vector(d$yb)), cor(pb, d$ya[, 1L]))
  expect_gt(cor(pb, as.vector(d$yb)), cor(pb, d$ya[, 2L]))
  expect_gt(cor(pb, as.vector(d$yb)), 0.9)

  # Likewise each of head_a's columns against the other's target.
  expect_gt(cor(preds[[1L]][, 1L], d$ya[, 1L]), cor(preds[[1L]][, 1L], d$ya[, 2L]))
  expect_gt(cor(preds[[1L]][, 2L], d$ya[, 2L]), cor(preds[[1L]][, 2L], d$ya[, 1L]))

  cleanup_mo_model(m)
})

test_that("per-head validation metrics are reported on the multi-input path", {
  d <- make_xy_multi_in()
  set.seed(1L)
  m <- build_two_in_two_head()
  m <- ggml_compile(m, loss = c("mse", "mse"), backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 4L, batch_size = 16L, validation_split = 0.25,
                verbose = 0L)
  h <- m$history

  expect_true(all(is.finite(h$val_loss)))
  expect_true(all(is.finite(h$val_head_a_loss)))
  expect_true(all(is.finite(h$val_head_b_loss)))

  cleanup_mo_model(m)
})

test_that("a callback on the multi-input path sees per-head keys", {
  d <- make_xy_multi_in()
  seen <- NULL
  spy <- list(on_epoch_end = function(epoch, logs, state) {
    seen <<- names(logs)
    invisible(NULL)
  })

  set.seed(1L)
  m <- build_two_in_two_head()
  m <- ggml_compile(m, loss = c("mse", "mse"), backend = "cpu")
  m <- ggml_fit(m, d$x, list(head_a = d$ya, head_b = d$yb),
                epochs = 2L, batch_size = 16L, verbose = 0L,
                callbacks = list(spy))

  expect_true(all(c("train_head_a_loss", "train_head_b_loss",
                    "val_head_a_loss", "val_head_b_loss") %in% seen))

  cleanup_mo_model(m)
})

test_that("a CE head and an MSE head mix on the multi-input path", {
  # Only the CE head loses its softmax; the MSE head keeps its activation. The
  # per-output logits flag has to survive onto this path too.
  set.seed(42L)
  n  <- 64L
  x1 <- matrix(runif(n * 4L), nrow = n)
  x2 <- matrix(runif(n * 3L), nrow = n)
  cls <- as.integer(rowSums(x1) + rowSums(x2) > 3.5)
  yc  <- cbind(1 - cls, cls)
  yr  <- matrix(rowSums(x1) + rowSums(x2), ncol = 1L)

  set.seed(1L)
  m <- build_two_in_two_head(units_a = 2L, units_b = 1L, act_a = "softmax")
  m <- ggml_compile(m, loss = c(head_a = "categorical_crossentropy", head_b = "mse"),
                    backend = "cpu")
  m <- ggml_fit(m, list(x1, x2), list(head_a = yc, head_b = yr),
                epochs = 8L, batch_size = 16L, verbose = 0L)
  h <- m$history

  expect_true(all(is.finite(h$train_head_a_loss)))
  expect_true(all(is.finite(h$train_head_b_loss)))
  # Accuracy is reported for the CE head only.
  expect_true(!is.null(h$train_head_a_accuracy))
  expect_true(all(is.finite(h$train_head_a_accuracy)))

  cleanup_mo_model(m)
})

# ---------------------------------------------------------------------------
# validation_data (as opposed to validation_split) on the multi-output path
# ---------------------------------------------------------------------------
# Regression guard: the training `y` is concatenated column-wise into one label
# matrix, but an explicit validation_data list used to be rbind()ed onto it
# unchanged. rbind(matrix, list) yields a *list-matrix* with the whole list
# folded into a single row, which then either aborts the batch-boundary
# truncation with "subscript out of bounds" or, when the row count happens to
# divide evenly by batch_size, trains on silently corrupted labels.

test_that("multi-output accepts validation_data as a list of matrices", {
  skip_on_cran()
  d  <- make_xy(n = 64L)
  dv <- make_xy(n = 32L)

  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, optimizer = "adam", loss = "mse", backend = "cpu")
  model <- ggml_fit(model, d$x, list(head_a = d$ya, head_b = d$yb),
                    epochs = 4L, batch_size = 16L,
                    validation_data = list(dv$x, list(head_a = dv$ya,
                                                      head_b = dv$yb)),
                    verbose = 0L)

  h <- model$history
  # Per-head validation keys exist and are real numbers, not NaN from a
  # mis-shaped label matrix.
  expect_true(!is.null(h$val_head_a_loss))
  expect_true(!is.null(h$val_head_b_loss))
  expect_true(all(is.finite(h$val_head_a_loss)))
  expect_true(all(is.finite(h$val_head_b_loss)))
  expect_true(all(is.finite(h$train_head_a_loss)))
  expect_true(all(is.finite(h$train_head_b_loss)))

  cleanup_mo_model(model)
})

test_that("validation_data survives a row count that needs truncation", {
  skip_on_cran()
  # 50 + 22 = 72 rows total; batch_size 16 leaves 8 rows over, so the
  # truncation branch -- where the list-matrix used to blow up -- is taken.
  d  <- make_xy(n = 50L)
  dv <- make_xy(n = 22L)
  expect_gt((50L + 22L) %% 16L, 0L)

  set.seed(1L)
  model <- build_two_head()
  model <- ggml_compile(model, optimizer = "adam", loss = "mse", backend = "cpu")
  model <- ggml_fit(model, d$x, list(head_a = d$ya, head_b = d$yb),
                    epochs = 3L, batch_size = 16L,
                    validation_data = list(dv$x, list(head_a = dv$ya,
                                                      head_b = dv$yb)),
                    verbose = 0L)

  h <- model$history
  expect_true(all(is.finite(h$train_head_a_loss)))
  expect_true(all(is.finite(h$val_head_a_loss)))

  cleanup_mo_model(model)
})

test_that("validation_data heads are matched by name, not by position", {
  skip_on_cran()
  d  <- make_xy(n = 64L)
  dv <- make_xy(n = 32L)

  # head_a is 2 columns and head_b is 1, so a positional read of a reversed
  # list would mis-slice the labels; by name the order must not matter.
  set.seed(1L)
  m1 <- build_two_head()
  m1 <- ggml_compile(m1, optimizer = "adam", loss = "mse", backend = "cpu")
  m1 <- ggml_fit(m1, d$x, list(head_a = d$ya, head_b = d$yb),
                 epochs = 3L, batch_size = 16L,
                 validation_data = list(dv$x, list(head_a = dv$ya,
                                                   head_b = dv$yb)),
                 verbose = 0L)

  set.seed(1L)
  m2 <- build_two_head()
  m2 <- ggml_compile(m2, optimizer = "adam", loss = "mse", backend = "cpu")
  m2 <- ggml_fit(m2, d$x, list(head_a = d$ya, head_b = d$yb),
                 epochs = 3L, batch_size = 16L,
                 validation_data = list(dv$x, list(head_b = dv$yb,
                                                   head_a = dv$ya)),
                 verbose = 0L)

  expect_equal(m1$history$val_head_a_loss, m2$history$val_head_a_loss,
               tolerance = 1e-6)
  expect_equal(m1$history$val_head_b_loss, m2$history$val_head_b_loss,
               tolerance = 1e-6)

  cleanup_mo_model(m1)
  cleanup_mo_model(m2)
})

test_that("malformed multi-output validation_data is rejected, not aborted", {
  skip_on_cran()
  d  <- make_xy(n = 64L)
  dv <- make_xy(n = 32L)

  fit_with <- function(y_val) {
    set.seed(1L)
    model <- build_two_head()
    model <- ggml_compile(model, optimizer = "adam", loss = "mse",
                          backend = "cpu")
    on.exit(cleanup_mo_model(model), add = TRUE)
    ggml_fit(model, d$x, list(head_a = d$ya, head_b = d$yb),
             epochs = 2L, batch_size = 16L,
             validation_data = list(dv$x, y_val), verbose = 0L)
  }

  # A bare matrix cannot say which head it belongs to.
  expect_error(fit_with(dv$ya), "must be a list")
  # Too few heads.
  expect_error(fit_with(list(dv$ya)), "one entry per output")
  # Right count, wrong widths: head_a is 2 columns and head_b is 1.
  expect_error(fit_with(list(dv$yb, dv$ya)), "widths")
  # Names that match no output.
  expect_error(fit_with(list(nope = dv$ya, head_b = dv$yb)),
               "do not match model outputs")
})
