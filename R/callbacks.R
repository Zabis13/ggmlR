# Callbacks for ggml_fit() epoch loop
# Each callback factory returns a list with on_train_begin, on_epoch_begin
# and/or on_epoch_end; every hook is optional.
#
# on_train_begin(logs, state) runs once per ggml_fit(), before the first epoch,
# with the same `state` environment the epoch hooks get. Stateful callbacks
# (early stopping, reduce-on-plateau, cosine decay) reset there, so one
# callback object can be reused across fits -- resample() folds, tune_grid()
# candidates -- each starting from scratch, as in Keras.

# Run the on_train_begin hooks. Called by every epoch loop (ggml_fit_opt,
# ggml_fit_opt_multi, ggml_fit_opt_gen and the two loops in the functional
# ggml_fit) right after `state` is built; each ggml_fit() goes through exactly
# one loop, so the hook fires once per fit.
.ggml_cb_train_begin <- function(callbacks, state) {
  for (cb in callbacks) {
    if (is.function(cb$on_train_begin)) cb$on_train_begin(list(), state)
  }
  invisible(NULL)
}

# ============================================================================
# Early Stopping
# ============================================================================

#' Early stopping callback
#'
#' Stops training when the monitored metric does not improve.
#'
#' @param monitor Metric to monitor: "val_loss", "val_accuracy", "train_loss", "train_accuracy"
#' @param patience Number of epochs with no improvement before stopping
#' @param min_delta Minimum change to qualify as improvement
#' @param mode "min" (lower is better) or "max" (higher is better). "auto" infers from monitor name.
#' @return List with on_train_begin and on_epoch_end functions; the state resets at the start of every fit
#' @export
#' @family callbacks
ggml_callback_early_stopping <- function(monitor = "val_loss", patience = 5,
                                          min_delta = 0, mode = "auto") {
  if (mode == "auto") {
    mode <- if (grepl("loss", monitor)) "min" else "max"
  }
  best <- if (mode == "min") Inf else -Inf
  wait <- 0L

  list(
    # every fit starts afresh: the object may be reused across fits
    on_train_begin = function(logs, state) {
      best <<- if (mode == "min") Inf else -Inf
      wait <<- 0L
      invisible(NULL)
    },
    on_epoch_end = function(epoch, logs, state) {
      val <- logs[[monitor]]
      if (is.null(val) || is.na(val)) return(invisible(NULL))

      improved <- if (mode == "min") val < best - min_delta else val > best + min_delta
      if (improved) {
        best <<- val
        wait <<- 0L
      } else {
        wait <<- wait + 1L
        if (wait >= patience) {
          message(sprintf("Early stopping at epoch %d (no improvement in %s for %d epochs)",
                          epoch, monitor, patience))
          state$stop <- TRUE
        }
      }
      invisible(NULL)
    }
  )
}

# ============================================================================
# LR Schedulers
# ============================================================================

#' Step decay LR scheduler
#'
#' Reduces LR by a factor every `step_size` epochs.
#'
#' @param step_size Reduce LR every this many epochs
#' @param gamma Multiplicative factor of LR reduction
#' @return List with on_epoch_begin function
#' @export
#' @family callbacks
ggml_schedule_step_decay <- function(step_size = 10, gamma = 0.1) {
  list(
    on_epoch_begin = function(epoch, logs, state) {
      if (epoch > 1 && (epoch - 1) %% step_size == 0) {
        lr <- .Call("R_ggml_opt_get_lr", state$lr_ud)
        new_lr <- lr["adamw"] * gamma
        .Call("R_ggml_opt_set_lr", state$lr_ud,
              as.numeric(new_lr), as.numeric(NA))
        message(sprintf("Epoch %d: LR reduced to %.6f", epoch, new_lr))
      }
      invisible(NULL)
    }
  )
}

#' Cosine annealing LR scheduler
#'
#' Anneals LR from initial value to `eta_min` following a cosine curve.
#'
#' @param eta_min Minimum LR at end of schedule
#' @param T_max Total number of epochs (defaults to nepoch from fit state)
#' @return List with on_train_begin and on_epoch_begin functions; eta_max is re-read from the LR at the start of every fit
#' @export
#' @family callbacks
ggml_schedule_cosine_decay <- function(eta_min = 0, T_max = NULL) {
  eta_max <- NULL  # captured at epoch 1 of each fit

  list(
    # eta_max is each fit's own starting LR, not the first fit's
    on_train_begin = function(logs, state) {
      eta_max <<- NULL
      invisible(NULL)
    },
    on_epoch_begin = function(epoch, logs, state) {
      if (is.null(eta_max)) {
        lr <- .Call("R_ggml_opt_get_lr", state$lr_ud)
        eta_max <<- lr["adamw"]
      }
      t_max <- if (!is.null(T_max)) T_max else state$nepoch
      new_lr <- eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * (epoch - 1) / t_max))
      .Call("R_ggml_opt_set_lr", state$lr_ud,
            as.numeric(new_lr), as.numeric(NA))
      invisible(NULL)
    }
  )
}

#' Reduce on plateau LR scheduler
#'
#' Reduces LR when a metric stops improving.
#'
#' @param monitor Metric to monitor: "val_loss", "train_loss", etc.
#' @param factor Factor to reduce LR by
#' @param patience Epochs with no improvement before reducing
#' @param min_lr Minimum LR
#' @param min_delta Minimum change to qualify as improvement
#' @param mode "min" or "max". "auto" infers from monitor name.
#' @return List with on_train_begin and on_epoch_end functions; the state resets at the start of every fit
#' @export
#' @family callbacks
ggml_schedule_reduce_on_plateau <- function(monitor = "val_loss", factor = 0.5,
                                             patience = 5, min_lr = 1e-7,
                                             min_delta = 1e-4, mode = "auto") {
  if (mode == "auto") {
    mode <- if (grepl("loss", monitor)) "min" else "max"
  }
  best <- if (mode == "min") Inf else -Inf
  wait <- 0L

  list(
    # every fit starts afresh: the object may be reused across fits
    on_train_begin = function(logs, state) {
      best <<- if (mode == "min") Inf else -Inf
      wait <<- 0L
      invisible(NULL)
    },
    on_epoch_end = function(epoch, logs, state) {
      val <- logs[[monitor]]
      if (is.null(val) || is.na(val)) return(invisible(NULL))

      improved <- if (mode == "min") val < best - min_delta else val > best + min_delta
      if (improved) {
        best <<- val
        wait <<- 0L
      } else {
        wait <<- wait + 1L
        if (wait >= patience) {
          lr <- .Call("R_ggml_opt_get_lr", state$lr_ud)
          new_lr <- max(lr["adamw"] * factor, min_lr)
          .Call("R_ggml_opt_set_lr", state$lr_ud,
                as.numeric(new_lr), as.numeric(NA))
          message(sprintf("Epoch %d: LR reduced to %.6f (plateau on %s)",
                          epoch, new_lr, monitor))
          wait <<- 0L
        }
      }
      invisible(NULL)
    }
  )
}
