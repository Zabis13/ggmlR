# Tests for callbacks and ggml_fit() R-side epoch loop

# ============================================================================
# Helper: build a minimal trainable linear regression setup
# Returns list(cpu, sched, ctx_compute, inputs, outputs, dataset)
# y = W*x, ndata=100, ne_in=4, ne_out=1, batch=10
# ============================================================================
make_linear_setup <- function(ndata = 100, ne_in = 4, ne_out = 1, batch = 10) {
  cpu  <- ggml_backend_cpu_init()
  sched <- ggml_backend_sched_new(list(cpu), parallel = FALSE)

  # Build static compute graph: out = x * W + b
  ctx_compute <- ggml_init_auto(4 * 1024 * 1024, no_alloc = TRUE)
  x_in  <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, ne_in,  batch)
  W     <- ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, ne_in,  ne_out)
  b     <- ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, ne_out)
  ggml_set_param(x_in)
  ggml_set_param(W)
  ggml_set_param(b)

  out <- ggml_add(ctx_compute,
                  ggml_mul_mat(ctx_compute, W, x_in),
                  b)

  # Allocate all tensors on CPU
  buf <- ggml_backend_alloc_ctx_tensors(ctx_compute, cpu)

  # Initialize weights to small random values
  ggml_backend_tensor_set_data(W, rnorm(ne_in * ne_out, sd = 0.1))
  ggml_backend_tensor_set_data(b, rep(0, ne_out))

  # Dataset: random X, y = X %*% c(1,2,3,4)
  true_w <- c(1, 2, 3, 4)[seq_len(ne_in)]
  X_all  <- matrix(rnorm(ndata * ne_in), nrow = ndata)
  y_all  <- X_all %*% true_w

  dataset <- ggml_opt_dataset_init(
    type_data     = GGML_TYPE_F32,
    type_label    = GGML_TYPE_F32,
    ne_datapoint  = ne_in,
    ne_label      = ne_out,
    ndata         = ndata,
    ndata_shard   = 1
  )
  ggml_backend_tensor_set_data(ggml_opt_dataset_data(dataset),   as.numeric(t(X_all)))
  ggml_backend_tensor_set_data(ggml_opt_dataset_labels(dataset), as.numeric(y_all))

  list(cpu = cpu, sched = sched, ctx_compute = ctx_compute,
       inputs = x_in, outputs = out, dataset = dataset, buf = buf)
}

cleanup_setup <- function(s) {
  ggml_opt_dataset_free(s$dataset)
  ggml_backend_buffer_free(s$buf)
  ggml_free(s$ctx_compute)
  ggml_backend_sched_free(s$sched)
  ggml_backend_free(s$cpu)
}

# ============================================================================
# ggml_opt_init_for_fit / ggml_opt_set_lr / ggml_opt_get_lr
# ============================================================================

test_that("ggml_opt_init_for_fit returns opt_ctx and lr_ud", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(
    sched        = s$sched,
    loss_type    = ggml_opt_loss_type_mse(),
    optimizer    = ggml_opt_optimizer_type_adamw(),
    opt_period   = 1L,
    ctx_compute  = s$ctx_compute,
    inputs       = s$inputs,
    outputs      = s$outputs
  )

  expect_type(ctx_list, "list")
  expect_true("opt_ctx" %in% names(ctx_list))
  expect_true("lr_ud"   %in% names(ctx_list))
  expect_false(is.null(ctx_list$opt_ctx))
  expect_false(is.null(ctx_list$lr_ud))

  ggml_opt_free(ctx_list$opt_ctx)
})

test_that("ggml_opt_get_lr returns named numeric vector", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  lr <- ggml_opt_get_lr(ctx_list$lr_ud)

  expect_type(lr, "double")
  expect_equal(length(lr), 2)
  expect_true("adamw" %in% names(lr))
  expect_true("sgd"   %in% names(lr))
  expect_gt(lr["adamw"], 0)
})

test_that("ggml_opt_set_lr updates AdamW LR", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  ggml_opt_set_lr(ctx_list$lr_ud, adamw_lr = 0.123)
  lr <- ggml_opt_get_lr(ctx_list$lr_ud)
  expect_equal(unname(lr["adamw"]), 0.123, tolerance = 1e-6)
})

test_that("ggml_opt_set_lr with NA does not change LR", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  lr_before <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]
  ggml_opt_set_lr(ctx_list$lr_ud, adamw_lr = NA)
  lr_after  <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]
  expect_equal(lr_before, lr_after)
})

# ============================================================================
# ggml_fit() — basic run
# ============================================================================

test_that("ggml_fit returns data frame with correct columns", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  hist <- ggml_fit(
    sched    = s$sched,
    ctx_compute = s$ctx_compute,
    inputs   = s$inputs,
    outputs  = s$outputs,
    dataset  = s$dataset,
    loss_type = ggml_opt_loss_type_mse(),
    nepoch   = 2L,
    nbatch_logical = 10L,
    val_split = 0.0,
    silent   = TRUE
  )

  expect_s3_class(hist, "data.frame")
  expect_true("epoch"          %in% names(hist))
  expect_true("train_loss"     %in% names(hist))
  expect_true("train_accuracy" %in% names(hist))
  expect_true("val_loss"       %in% names(hist))
  expect_true("val_accuracy"   %in% names(hist))
  expect_equal(nrow(hist), 2)
})

test_that("ggml_fit epoch column is sequential", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  hist <- ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
                   nepoch = 3L, nbatch_logical = 10L, silent = TRUE)

  expect_equal(hist$epoch, 1:3)
})

test_that("ggml_fit val_loss is NA when val_split=0", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  hist <- ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
                   nepoch = 2L, nbatch_logical = 10L, val_split = 0.0, silent = TRUE)

  expect_true(all(is.na(hist$val_loss)))
  expect_true(all(is.na(hist$val_accuracy)))
})

test_that("ggml_fit val_loss is numeric when val_split>0", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  hist <- ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
                   nepoch = 2L, nbatch_logical = 10L, val_split = 0.2, silent = TRUE)

  expect_true(all(!is.na(hist$val_loss)))
  expect_true(all(is.finite(hist$val_loss)))
})

test_that("ggml_fit train_loss is finite positive", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  hist <- ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
                   nepoch = 3L, nbatch_logical = 10L, silent = TRUE)

  expect_true(all(is.finite(hist$train_loss)))
  expect_true(all(hist$train_loss >= 0))
})

test_that("ggml_fit with callbacks=list() works (no callbacks)", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  expect_no_error(
    ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
             nepoch = 2L, nbatch_logical = 10L,
             callbacks = list(), silent = TRUE)
  )
})

# ============================================================================
# ggml_callback_early_stopping
# ============================================================================

test_that("ggml_callback_early_stopping returns list with on_epoch_end", {
  cb <- ggml_callback_early_stopping()
  expect_type(cb, "list")
  expect_true("on_epoch_end" %in% names(cb))
  expect_true(is.function(cb$on_epoch_end))
})

test_that("ggml_callback_early_stopping default args", {
  cb <- ggml_callback_early_stopping()
  # Just check it constructs without error
  expect_type(cb, "list")
})

test_that("ggml_callback_early_stopping sets state$stop after patience", {
  cb <- ggml_callback_early_stopping(monitor = "val_loss", patience = 2, min_delta = 0)

  state <- new.env(parent = emptyenv())
  state$stop <- FALSE

  # Epoch 1: first value — sets best, no stop
  suppressMessages(cb$on_epoch_end(1, list(val_loss = 1.0), state))
  expect_false(state$stop)

  # Epoch 2: no improvement (wait=1, patience=2)
  suppressMessages(cb$on_epoch_end(2, list(val_loss = 1.0), state))
  expect_false(state$stop)

  # Epoch 3: no improvement (wait=2 >= patience=2) → stop
  suppressMessages(cb$on_epoch_end(3, list(val_loss = 1.0), state))
  expect_true(state$stop)
})

test_that("ggml_callback_early_stopping resets wait on improvement", {
  cb <- ggml_callback_early_stopping(monitor = "val_loss", patience = 2)
  state <- new.env(parent = emptyenv())
  state$stop <- FALSE

  suppressMessages(cb$on_epoch_end(1, list(val_loss = 1.0), state))
  suppressMessages(cb$on_epoch_end(2, list(val_loss = 1.0), state))  # wait=1
  suppressMessages(cb$on_epoch_end(3, list(val_loss = 0.5), state))  # improvement → wait=0
  expect_false(state$stop)

  suppressMessages(cb$on_epoch_end(4, list(val_loss = 0.5), state))  # wait=1
  expect_false(state$stop)
})

test_that("ggml_callback_early_stopping mode=max works", {
  cb <- ggml_callback_early_stopping(monitor = "train_accuracy", patience = 1, mode = "max")
  state <- new.env(parent = emptyenv())
  state$stop <- FALSE

  suppressMessages(cb$on_epoch_end(1, list(train_accuracy = 0.8), state))
  suppressMessages(cb$on_epoch_end(2, list(train_accuracy = 0.7), state))  # wait=1 >= patience=1 → stop
  expect_true(state$stop)
})

test_that("ggml_callback_early_stopping ignores NA metric", {
  cb <- ggml_callback_early_stopping(monitor = "val_loss", patience = 1)
  state <- new.env(parent = emptyenv())
  state$stop <- FALSE

  cb$on_epoch_end(1, list(val_loss = NA), state)
  expect_false(state$stop)
})

test_that("ggml_callback_early_stopping stops ggml_fit early", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  # patience=1: should stop after 2 epochs with no improvement
  hist <- suppressMessages(ggml_fit(
    s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
    nepoch = 20L, nbatch_logical = 10L,
    val_split = 0.2, silent = TRUE,
    callbacks = list(
      ggml_callback_early_stopping(monitor = "val_loss", patience = 1)
    )
  ))

  # Should have stopped before 20 epochs
  expect_lt(nrow(hist), 20)
})

# ============================================================================
# ggml_schedule_step_decay
# ============================================================================

test_that("ggml_schedule_step_decay returns list with on_epoch_begin", {
  cb <- ggml_schedule_step_decay()
  expect_type(cb, "list")
  expect_true("on_epoch_begin" %in% names(cb))
  expect_true(is.function(cb$on_epoch_begin))
})

test_that("ggml_schedule_step_decay reduces LR at step boundary", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  initial_lr <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]

  state <- new.env(parent = emptyenv())
  state$stop  <- FALSE
  state$lr_ud <- ctx_list$lr_ud

  cb <- ggml_schedule_step_decay(step_size = 2, gamma = 0.5)

  # Epoch 1: no reduction
  suppressMessages(cb$on_epoch_begin(1, list(), state))
  expect_equal(ggml_opt_get_lr(ctx_list$lr_ud)["adamw"], initial_lr, tolerance = 1e-6)

  # Epoch 2: no reduction (step boundary is at epoch 3: (3-1) %% 2 == 0)
  suppressMessages(cb$on_epoch_begin(2, list(), state))
  expect_equal(ggml_opt_get_lr(ctx_list$lr_ud)["adamw"], initial_lr, tolerance = 1e-6)

  # Epoch 3: (3-1) %% 2 == 0 → reduce
  suppressMessages(cb$on_epoch_begin(3, list(), state))
  expect_equal(ggml_opt_get_lr(ctx_list$lr_ud)["adamw"],
               initial_lr * 0.5, tolerance = 1e-6)
})

# ============================================================================
# ggml_schedule_cosine_decay
# ============================================================================

test_that("ggml_schedule_cosine_decay returns list with on_epoch_begin", {
  cb <- ggml_schedule_cosine_decay()
  expect_type(cb, "list")
  expect_true("on_epoch_begin" %in% names(cb))
})

test_that("ggml_schedule_cosine_decay decreases LR over epochs", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  state <- new.env(parent = emptyenv())
  state$stop   <- FALSE
  state$lr_ud  <- ctx_list$lr_ud
  state$nepoch <- 10L

  cb <- ggml_schedule_cosine_decay(eta_min = 0, T_max = 10)

  lrs <- numeric(10)
  for (ep in 1:10) {
    cb$on_epoch_begin(ep, list(), state)
    lrs[ep] <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]
  }

  # LR at epoch 1 should be max (or near it), at epoch 10 near 0
  expect_gt(lrs[1], lrs[10])
  expect_gte(lrs[10], 0)
})

test_that("ggml_schedule_cosine_decay with eta_min > 0 stays above eta_min", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  state <- new.env(parent = emptyenv())
  state$stop   <- FALSE
  state$lr_ud  <- ctx_list$lr_ud
  state$nepoch <- 5L

  eta_min <- 1e-5
  cb <- ggml_schedule_cosine_decay(eta_min = eta_min, T_max = 5)

  for (ep in 1:5) cb$on_epoch_begin(ep, list(), state)

  lr_final <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]
  expect_gte(lr_final, eta_min - 1e-8)
})

# ============================================================================
# ggml_schedule_reduce_on_plateau
# ============================================================================

test_that("ggml_schedule_reduce_on_plateau returns list with on_epoch_end", {
  cb <- ggml_schedule_reduce_on_plateau()
  expect_type(cb, "list")
  expect_true("on_epoch_end" %in% names(cb))
})

test_that("ggml_schedule_reduce_on_plateau reduces LR after patience epochs", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  initial_lr <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]

  state <- new.env(parent = emptyenv())
  state$stop  <- FALSE
  state$lr_ud <- ctx_list$lr_ud

  cb <- ggml_schedule_reduce_on_plateau(monitor = "val_loss", factor = 0.5,
                                         patience = 2, min_lr = 1e-9)

  suppressMessages(cb$on_epoch_end(1, list(val_loss = 1.0), state))  # best=1.0, wait=0
  suppressMessages(cb$on_epoch_end(2, list(val_loss = 1.0), state))  # wait=1
  suppressMessages(cb$on_epoch_end(3, list(val_loss = 1.0), state))  # wait=2 >= patience → reduce

  lr_new <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]
  expect_equal(lr_new, initial_lr * 0.5, tolerance = 1e-6)
})

test_that("ggml_schedule_reduce_on_plateau respects min_lr floor", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  # Set very small initial LR
  ggml_opt_set_lr(ctx_list$lr_ud, adamw_lr = 1e-8)

  state <- new.env(parent = emptyenv())
  state$stop  <- FALSE
  state$lr_ud <- ctx_list$lr_ud

  min_lr <- 1e-7
  cb <- ggml_schedule_reduce_on_plateau(monitor = "val_loss", factor = 0.1,
                                         patience = 1, min_lr = min_lr)

  suppressMessages(cb$on_epoch_end(1, list(val_loss = 1.0), state))
  suppressMessages(cb$on_epoch_end(2, list(val_loss = 1.0), state))  # reduce: max(1e-8 * 0.1, 1e-7) = 1e-7

  lr_new <- ggml_opt_get_lr(ctx_list$lr_ud)["adamw"]
  expect_gte(lr_new, min_lr - 1e-10)
})

test_that("ggml_schedule_reduce_on_plateau resets wait after reduction", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  ctx_list <- ggml_opt_init_for_fit(s$sched, ggml_opt_loss_type_mse(),
                                     ggml_opt_optimizer_type_adamw(), 1L,
                                     s$ctx_compute, s$inputs, s$outputs)
  on.exit(ggml_opt_free(ctx_list$opt_ctx), add = TRUE)

  initial_lr <- unname(ggml_opt_get_lr(ctx_list$lr_ud)["adamw"])

  state <- new.env(parent = emptyenv())
  state$stop  <- FALSE
  state$lr_ud <- ctx_list$lr_ud

  # patience=2: reduce after 2 consecutive non-improving epochs, then wait resets
  cb <- ggml_schedule_reduce_on_plateau(monitor = "val_loss", factor = 0.5,
                                         patience = 2, min_lr = 1e-9)

  suppressMessages(cb$on_epoch_end(1, list(val_loss = 1.0), state))  # best=1.0, wait=0
  suppressMessages(cb$on_epoch_end(2, list(val_loss = 1.0), state))  # wait=1
  suppressMessages(cb$on_epoch_end(3, list(val_loss = 1.0), state))  # wait=2 >= 2 → reduce, wait=0
  lr_after_first_reduce <- unname(ggml_opt_get_lr(ctx_list$lr_ud)["adamw"])
  expect_equal(lr_after_first_reduce, initial_lr * 0.5, tolerance = 1e-6)

  # After reset, wait=0: next 2 non-improving epochs needed before second reduce
  suppressMessages(cb$on_epoch_end(4, list(val_loss = 1.0), state))  # wait=1
  lr_after_fourth <- unname(ggml_opt_get_lr(ctx_list$lr_ud)["adamw"])
  expect_equal(lr_after_fourth, lr_after_first_reduce, tolerance = 1e-8)  # no reduction yet

  suppressMessages(cb$on_epoch_end(5, list(val_loss = 1.0), state))  # wait=2 >= 2 → second reduce
  lr_after_second_reduce <- unname(ggml_opt_get_lr(ctx_list$lr_ud)["adamw"])
  expect_equal(lr_after_second_reduce, lr_after_first_reduce * 0.5, tolerance = 1e-7)
})

# ============================================================================
# ggml_fit() + callbacks integration
# ============================================================================

test_that("ggml_fit with step_decay callback runs without error", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  expect_no_error(suppressMessages(
    ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
             nepoch = 4L, nbatch_logical = 10L, silent = TRUE,
             callbacks = list(ggml_schedule_step_decay(step_size = 2, gamma = 0.5)))
  ))
})

test_that("ggml_fit with cosine_decay callback runs without error", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  expect_no_error(suppressMessages(
    ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
             nepoch = 4L, nbatch_logical = 10L, silent = TRUE,
             callbacks = list(ggml_schedule_cosine_decay(T_max = 4)))
  ))
})

test_that("ggml_fit with reduce_on_plateau callback runs without error", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  expect_no_error(suppressMessages(
    ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
             nepoch = 4L, nbatch_logical = 10L, val_split = 0.2, silent = TRUE,
             callbacks = list(ggml_schedule_reduce_on_plateau(patience = 2)))
  ))
})

test_that("ggml_fit with multiple callbacks runs without error", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  expect_no_error(suppressMessages(
    ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
             nepoch = 6L, nbatch_logical = 10L, val_split = 0.2, silent = TRUE,
             callbacks = list(
               ggml_schedule_cosine_decay(T_max = 6),
               ggml_callback_early_stopping(monitor = "val_loss", patience = 3)
             ))
  ))
})

test_that("custom on_epoch_end callback receives correct epoch number", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  observed_epochs <- integer(0)
  custom_cb <- list(
    on_epoch_end = function(epoch, logs, state) {
      observed_epochs <<- c(observed_epochs, epoch)
    }
  )

  ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
           nepoch = 3L, nbatch_logical = 10L, silent = TRUE,
           callbacks = list(custom_cb))

  expect_equal(observed_epochs, 1:3)
})

test_that("custom on_epoch_begin callback receives correct epoch number", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  observed_epochs <- integer(0)
  custom_cb <- list(
    on_epoch_begin = function(epoch, logs, state) {
      observed_epochs <<- c(observed_epochs, epoch)
    }
  )

  ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
           nepoch = 3L, nbatch_logical = 10L, silent = TRUE,
           callbacks = list(custom_cb))

  expect_equal(observed_epochs, 1:3)
})

test_that("custom callback logs contain train_loss", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  captured_logs <- list()
  custom_cb <- list(
    on_epoch_end = function(epoch, logs, state) {
      captured_logs[[epoch]] <<- logs
    }
  )

  ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
           nepoch = 2L, nbatch_logical = 10L, silent = TRUE,
           callbacks = list(custom_cb))

  expect_equal(length(captured_logs), 2)
  expect_true("train_loss" %in% names(captured_logs[[1]]))
  expect_true(is.finite(captured_logs[[1]]$train_loss))
})

test_that("callback state$stop stops training", {
  s <- make_linear_setup()
  on.exit(cleanup_setup(s))

  stop_cb <- list(
    on_epoch_end = function(epoch, logs, state) {
      if (epoch >= 2) state$stop <- TRUE
    }
  )

  hist <- ggml_fit(s$sched, s$ctx_compute, s$inputs, s$outputs, s$dataset,
                   nepoch = 10L, nbatch_logical = 10L, silent = TRUE,
                   callbacks = list(stop_cb))

  expect_equal(nrow(hist), 2)
})
