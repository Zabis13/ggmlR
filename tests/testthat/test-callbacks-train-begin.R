# on_train_begin: stateful callbacks start every ggml_fit() from scratch.
#
# Early stopping and reduce-on-plateau kept best/wait in their closure, and
# cosine decay its eta_max, across fits: reusing one callback object for a
# second fit (resample() folds, tune_grid() candidates, a plain second
# ggml_fit) began where the previous fit ended -- a fold stopped at epoch 1
# or 2 instead of running. Each training loop now calls on_train_begin(logs,
# state) once before the first epoch (as Keras does); the three callbacks
# reset there. The hook is optional: callbacks without it behave as before.
#
# The five epoch loops are disjoint branches -- every ggml_fit() goes through
# exactly one: ggml_fit_opt (sequential, single-head functional),
# ggml_fit_opt_multi (multi-head), the functional dropout loop, the
# multi-input functional loop, and ggml_fit_opt_gen (generator).

ggml_set_n_threads(2L)

free_model <- function(model) {
  comp <- model$compilation
  if (!is.null(comp$sched))       ggml_backend_sched_free(comp$sched)
  if (!is.null(comp$backend))     ggml_backend_free(comp$backend)
  if (!is.null(comp$cpu_backend)) ggml_backend_free(comp$cpu_backend)
}

N <- 64L
make_x <- function(nfeat = 4L, seed = 1L) {
  set.seed(seed)
  matrix(runif(N * nfeat), nrow = N)
}
onehot <- function(x) {
  y <- matrix(0, nrow = nrow(x), ncol = 2L)
  y[cbind(seq_len(nrow(x)), ifelse(rowSums(x) > ncol(x) / 2, 1L, 2L))] <- 1
  y
}

seq_model <- function() {
  m <- ggml_model_sequential() |>
    ggml_layer_dense(8, activation = "relu") |>
    ggml_layer_dense(2, activation = "softmax")
  m$input_shape <- 4L
  ggml_compile(m, optimizer = "adam", loss = "categorical_crossentropy",
               backend = "cpu")
}

# fit() runs one fresh sequential model with the given callbacks
fit_seq <- function(callbacks, epochs = 3L) {
  set.seed(2L)
  m <- seq_model()
  x <- make_x()
  m <- ggml_fit(m, x, onehot(x), epochs = epochs, batch_size = 16L,
                verbose = 0L, callbacks = callbacks)
  free_model(m)
  invisible(m)
}

# ---------------------------------------------------------------------------
# 1. The three stateful callbacks reset between fits
# ---------------------------------------------------------------------------

# reads the callback's closure state at the first epoch of each fit; placed
# after the callback in the list, so it sees the state that fit starts from
probe_state <- function(cb, fn = "on_epoch_end") {
  env <- environment(cb[[fn]])
  seen <- list()
  list(
    probe = list(on_epoch_begin = function(epoch, logs, state) {
      if (epoch == 1L) seen[[length(seen) + 1L]] <<- list(best = env$best,
                                                         wait = env$wait)
      invisible(NULL)
    }),
    seen = function() seen
  )
}

test_that("early stopping starts every fit with best = Inf, wait = 0", {
  es <- ggml_callback_early_stopping(monitor = "train_loss", patience = 100)
  p  <- probe_state(es)
  fit_seq(list(es, p$probe))
  fit_seq(list(es, p$probe))
  s <- p$seen()
  expect_length(s, 2L)
  expect_identical(s[[1]]$best, Inf)
  expect_identical(s[[2]]$best, Inf)   # was the first fit's final loss
  expect_identical(s[[2]]$wait, 0L)
})

test_that("early stopping with patience no longer cuts the next fit short", {
  # first fit drives best low; the second (fresh weights) starts higher, and
  # with the old carried-over best it stopped after `patience` epochs
  es <- ggml_callback_early_stopping(monitor = "train_loss", patience = 2)
  suppressMessages(fit_seq(list(es), epochs = 8L))
  m2 <- suppressMessages(fit_seq(list(es), epochs = 3L))
  expect_length(m2$history$train_loss, 3L)
})

test_that("reduce-on-plateau starts every fit with best = Inf, wait = 0", {
  rp <- ggml_schedule_reduce_on_plateau(monitor = "train_loss", patience = 100)
  p  <- probe_state(rp)
  fit_seq(list(rp, p$probe))
  fit_seq(list(rp, p$probe))
  s <- p$seen()
  expect_identical(s[[2]]$best, Inf)
  expect_identical(s[[2]]$wait, 0L)
})

test_that("cosine decay anneals from each fit's own starting LR", {
  # set_lr runs first at epoch 1 and gives each fit a different LR; cosine
  # must take its eta_max from that, not from the first fit it ever saw
  start_lr <- NULL
  set_lr <- list(on_epoch_begin = function(epoch, logs, state) {
    if (epoch == 1L) .Call("R_ggml_opt_set_lr", state$lr_ud,
                           as.numeric(start_lr), as.numeric(NA))
    invisible(NULL)
  })
  cos <- ggml_schedule_cosine_decay(eta_min = 0)
  lr_seen <- numeric(0)
  read_lr <- list(on_epoch_begin = function(epoch, logs, state) {
    if (epoch == 1L) lr_seen <<- c(lr_seen,
                                   .Call("R_ggml_opt_get_lr", state$lr_ud)[["adamw"]])
    invisible(NULL)
  })

  start_lr <- 0.01
  fit_seq(list(set_lr, cos, read_lr))
  start_lr <- 0.05
  fit_seq(list(set_lr, cos, read_lr))

  # epoch 1 of a cosine schedule is eta_max itself
  expect_equal(lr_seen, c(0.01, 0.05), tolerance = 1e-6)
})

# ---------------------------------------------------------------------------
# 2. Hook contract on all five loops: once per fit, before the first epoch,
#    (logs, state) like on_epoch_begin; callbacks without it unaffected
# ---------------------------------------------------------------------------

spy <- function() {
  ev <- character(0)
  st <- NULL
  list(
    cb = list(
      on_train_begin = function(logs, state) {
        ev <<- c(ev, "train_begin")
        st <<- list(is_list = is.list(logs),
                    has_lr  = !is.null(state$lr_ud),
                    nepoch  = state$nepoch,
                    stop    = state$stop)
        invisible(NULL)
      },
      on_epoch_begin = function(epoch, logs, state) {
        ev <<- c(ev, paste0("epoch_", epoch)); invisible(NULL)
      }
    ),
    events = function() ev,
    state  = function() st
  )
}

# a callback written before on_train_begin existed
legacy <- function() {
  n <- 0L
  list(cb = list(on_epoch_end = function(epoch, logs, state) {
         n <<- n + 1L; invisible(NULL)
       }),
       count = function() n)
}

paths <- list(
  sequential = function(cbs) fit_seq(cbs, epochs = 3L),

  functional_single = function(cbs) {
    set.seed(3L)
    inp <- ggml_input(shape = 4L)
    out <- inp |> ggml_layer_dense(8L, activation = "relu") |>
      ggml_layer_dense(2L, activation = "softmax")
    m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                      loss = "categorical_crossentropy", backend = "cpu")
    x <- make_x()
    m <- ggml_fit(m, x, onehot(x), epochs = 3L, batch_size = 16L,
                  verbose = 0L, callbacks = cbs)
    free_model(m)
  },

  functional_multi_head = function(cbs) {
    set.seed(4L)
    inp   <- ggml_input(shape = 4L)
    trunk <- inp |> ggml_layer_dense(8L, activation = "relu")
    outa  <- ggml_apply(trunk, ggml_dense(2L, name = "head_a"))
    outb  <- ggml_apply(trunk, ggml_dense(1L, name = "head_b"))
    m <- ggml_compile(ggml_model(inputs = inp, outputs = list(outa, outb)),
                      loss = "mse", backend = "cpu")
    x <- make_x()
    m <- ggml_fit(m, x, list(head_a = cbind(x[, 1], x[, 2]),
                             head_b = matrix(rowSums(x))),
                  epochs = 3L, batch_size = 16L, verbose = 0L, callbacks = cbs)
    free_model(m)
  },

  functional_dropout = function(cbs) {
    set.seed(5L)
    inp <- ggml_input(shape = 4L)
    out <- inp |> ggml_layer_dense(8L, activation = "relu") |>
      ggml_layer_dropout(0.3, stochastic = TRUE) |>
      ggml_layer_dense(2L, activation = "softmax")
    m <- ggml_compile(ggml_model(inputs = inp, outputs = out),
                      loss = "categorical_crossentropy", backend = "cpu")
    x <- make_x()
    m <- ggml_fit(m, x, onehot(x), epochs = 3L, batch_size = 16L,
                  verbose = 0L, callbacks = cbs)
    free_model(m)
  },

  functional_multi_input = function(cbs) {
    set.seed(6L)
    inp1 <- ggml_input(shape = 4L)
    inp2 <- ggml_input(shape = 3L)
    br1  <- inp1 |> ggml_layer_dense(4L, activation = "relu")
    br2  <- inp2 |> ggml_layer_dense(4L, activation = "relu")
    out  <- ggml_layer_concatenate(list(br1, br2), axis = 0L) |>
      ggml_layer_dense(2L, activation = "softmax")
    m <- ggml_compile(ggml_model(inputs = list(inp1, inp2), outputs = out),
                      loss = "categorical_crossentropy", backend = "cpu")
    x1 <- make_x(4L); x2 <- make_x(3L, seed = 7L)
    m <- ggml_fit(m, list(x1, x2), onehot(x1), epochs = 3L, batch_size = 16L,
                  verbose = 0L, callbacks = cbs)
    free_model(m)
  },

  generator = function(cbs) {
    set.seed(8L)
    m <- seq_model()
    x <- make_x(); y <- onehot(x)
    i <- 0L
    gen <- function() {
      if (i >= 4L) { i <<- 0L; return(NULL) }
      idx <- (i * 16L + 1L):((i + 1L) * 16L)
      i <<- i + 1L
      list(x[idx, , drop = FALSE], y[idx, , drop = FALSE])
    }
    m <- ggml_fit(m, generator = gen, epochs = 3L, batch_size = 16L,
                  verbose = 0L, callbacks = cbs)
    free_model(m)
  }
)

for (nm in names(paths)) {
  local({
    path <- nm
    test_that(sprintf("on_train_begin fires once, before epoch 1 (%s)", path), {
      s <- spy(); l <- legacy()
      suppressMessages(paths[[path]](list(l$cb, s$cb)))
      ev <- s$events()
      expect_identical(ev, c("train_begin", "epoch_1", "epoch_2", "epoch_3"))
      st <- s$state()
      expect_true(st$is_list)
      expect_true(st$has_lr)
      expect_equal(st$nepoch, 3)
      expect_false(st$stop)
      # the legacy callback (no on_train_begin) still runs every epoch
      expect_identical(l$count(), 3L)
    })
  })
}
