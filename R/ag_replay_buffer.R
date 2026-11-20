# Replay buffer for off-policy RL: storage (one ag_buffer on the training
# device) composed with a sampler (uniform, or ag_per_sampler() when
# prioritized). The layout and file format follow rltoolsR's
# R/replay_buffer.R, whose files ag_replay_buffer_load() also reads.

AG_RB_FIELDS <- c("obs", "act", "reward", "terminated", "truncated", "next_obs")
AG_RB_FORMAT <- "ggmlR.replay_buffer.v1"
AG_RB_FORMAT_RLTOOLS <- "rltoolsR.replay_buffer.v1"

# Row ranges (0-based `from`, length `n`) of the packed transition.
.ag_rb_layout <- function(obs_dim, act_dim) {
  n <- c(obs = obs_dim, act = act_dim, reward = 1L, terminated = 1L,
         truncated = 1L, next_obs = obs_dim)
  from <- c(0L, cumsum(n)[-length(n)])
  names(from) <- names(n)
  list(from = from, n = n, rows = sum(n))
}

#' Replay buffer
#'
#' A ring of \code{capacity} transitions held in one \code{\link{ag_buffer}}
#' on the current device, one column per transition, packed as
#' \code{[obs | act | reward | terminated | truncated | next_obs]}, so a batch is
#' one \code{\link{ag_get_rows}} and its fields are row slices (\code{obs} and
#' \code{act} are adjacent: the critic input \eqn{Q(s, a)} is one slice).
#' \code{next_obs} must be the FINAL observation of an episode on automatic
#' reset, not the first observation of the next one.
#'
#' Methods:
#' \describe{
#'   \item{\code{$add(obs, act, reward, terminated, truncated, next_obs)}}{One
#'     step of \code{n_envs} transitions: \code{obs}, \code{next_obs}
#'     \code{[n_envs, obs_dim]}, \code{act} \code{[n_envs, act_dim]}, the rest
#'     length \code{n_envs}. Prioritized: the written slots get the largest
#'     priority seen so far.}
#'   \item{\code{$sample(batch, beta = 0.4)}}{\code{list(idx, weights)}: 0-based
#'     slots from the filled part and importance weights (all 1 when uniform).
#'     Uniform draws use \code{sample.int} (R's RNG, with replacement).}
#'   \item{\code{$gather(idx)}}{The batch as one packed ag_tensor
#'     \code{[rows, B]}. Inside \code{\link{ag_capture}} pass \code{idx} as an
#'     argument and list \code{rb$buf} in \code{params}.}
#'   \item{\code{$field(x, field)}, \code{$slice(x, fields)}}{One field, or
#'     adjacent fields, of a gathered batch.}
#'   \item{\code{$update_priorities(idx, td_errors)}}{Prioritized only.}
#'   \item{\code{$save(path)}, \code{$free()}, \code{$full()}}{}
#' }
#' \code{episode_start} records the slot where each transition's episode
#' began, for sequential batches.
#'
#' @param obs_dim,act_dim Observation and action sizes.
#' @param capacity Number of transitions, at least \code{n_envs}.
#' @param n_envs Transitions added per \code{$add()}.
#' @param prioritized Use \code{\link{ag_per_sampler}} (default \code{FALSE}).
#' @param alpha,eps Passed to \code{\link{ag_per_sampler}}.
#' @return An \code{ag_replay_buffer} environment.
#' @seealso \code{\link{ag_replay_buffer_load}}, \code{\link{ag_per_sampler}},
#'   \code{\link{ag_nstep_targets}}
#' @export
#' @examples
#' \donttest{
#' rb <- ag_replay_buffer(obs_dim = 4L, act_dim = 1L, capacity = 1000L,
#'                        prioritized = TRUE)
#' rb$add(matrix(0, 1, 4), matrix(1, 1, 1), 1, FALSE, FALSE, matrix(0, 1, 4))
#' b <- rb$sample(32L, beta = 0.4)
#' x <- rb$gather(b$idx)
#' r <- rb$field(x, "reward")
#' rb$update_priorities(b$idx, td_errors = rep(0.5, 32))
#' rb$free()
#' }
ag_replay_buffer <- function(obs_dim, act_dim, capacity, n_envs = 1L,
                             prioritized = FALSE, alpha = 0.6, eps = 1e-6) {
  obs_dim <- as.integer(obs_dim); act_dim <- as.integer(act_dim)
  capacity <- as.integer(capacity); n_envs <- as.integer(n_envs)
  if (anyNA(c(obs_dim, act_dim, capacity, n_envs)) || obs_dim < 1L || act_dim < 1L ||
      n_envs < 1L)
    stop("ag_replay_buffer: dimensions must be positive integers", call. = FALSE)
  if (capacity < n_envs)
    stop("ag_replay_buffer: capacity (", capacity, ") must be at least the number ",
         "of environments (", n_envs, ")", call. = FALSE)
  layout <- .ag_rb_layout(obs_dim, act_dim)
  rb <- new.env(parent = emptyenv())
  rb$obs_dim  <- obs_dim
  rb$act_dim  <- act_dim
  rb$capacity <- capacity
  rb$n_envs   <- n_envs
  rb$layout   <- layout
  rb$buf      <- ag_buffer(layout$rows, capacity)
  rb$pos      <- 0L                             # next slot to write
  rb$size     <- 0L                             # filled slots
  rb$episode_start <- rep(NA_integer_, capacity)
  rb$cur_start     <- rep(NA_integer_, n_envs)  # slot where each copy's episode began
  rb$sampler  <- if (isTRUE(prioritized)) ag_per_sampler(capacity, alpha, eps)

  rb$full <- function() rb$size == rb$capacity

  rb$add <- function(obs, act, reward, terminated, truncated, next_obs) {
    n <- rb$n_envs
    obs <- matrix(obs, n); act <- matrix(act, n); next_obs <- matrix(next_obs, n)
    if (ncol(obs) != rb$obs_dim || ncol(next_obs) != rb$obs_dim ||
        ncol(act) != rb$act_dim || length(reward) != n || length(terminated) != n ||
        length(truncated) != n)
      stop("ag_replay_buffer: transition shapes do not match (", n, " environments, ",
           "obs_dim ", rb$obs_dim, ", act_dim ", rb$act_dim, ")", call. = FALSE)
    data <- rbind(t(obs), t(act), as.numeric(reward), as.numeric(terminated),
                  as.numeric(truncated), t(next_obs))
    # ring write: two writes when the step crosses the end
    first <- min(n, rb$capacity - rb$pos)
    ag_buffer_write(rb$buf, data[, seq_len(first), drop = FALSE], rb$pos)
    if (first < n) ag_buffer_write(rb$buf, data[, -seq_len(first), drop = FALSE], 0L)

    slots <- (rb$pos + seq_len(n) - 1L) %% rb$capacity
    new_ep <- is.na(rb$cur_start)
    rb$cur_start[new_ep] <- slots[new_ep]
    rb$episode_start[slots + 1L] <- rb$cur_start
    rb$cur_start[as.logical(terminated) | as.logical(truncated)] <- NA_integer_
    if (!is.null(rb$sampler)) rb$sampler$add(slots)

    rb$pos  <- (rb$pos + n) %% rb$capacity
    rb$size <- min(rb$size + n, rb$capacity)
    invisible(rb)
  }

  rb$sample <- function(batch, beta = 0.4) {
    if (rb$size < 1L) stop("ag_replay_buffer: sampling from an empty buffer", call. = FALSE)
    batch <- as.integer(batch)
    if (!is.null(rb$sampler)) return(rb$sampler$sample(batch, beta))
    list(idx = sample.int(rb$size, batch, replace = TRUE) - 1L, weights = rep(1, batch))
  }

  rb$update_priorities <- function(idx, td_errors) {
    if (is.null(rb$sampler))
      stop("ag_replay_buffer: not prioritized (prioritized = FALSE)", call. = FALSE)
    rb$sampler$update(idx, td_errors)
  }

  rb$gather <- function(idx) ag_get_rows(rb$buf, idx)

  rb$field <- function(x, field) {
    field <- match.arg(field, AG_RB_FIELDS)
    ag_slice_rows(x, rb$layout$from[[field]], rb$layout$n[[field]])
  }

  rb$slice <- function(x, fields) {
    fields <- match.arg(fields, AG_RB_FIELDS, several.ok = TRUE)
    i <- match(fields, AG_RB_FIELDS)
    if (any(diff(sort(i)) != 1L))
      stop("ag_replay_buffer: fields ", paste(fields, collapse = ", "),
           " are not adjacent in the layout", call. = FALSE)
    first <- AG_RB_FIELDS[min(i)]
    ag_slice_rows(x, rb$layout$from[[first]], sum(rb$layout$n[fields]))
  }

  rb$free <- function() {
    if (!is.null(rb$buf)) ag_buffer_free(rb$buf)
    rb$buf <- NULL
    invisible(NULL)
  }

  # A serialized header (dimensions, position, size, episode bookkeeping,
  # priorities), then the filled slots as f32 in parts of `chunk` columns, so
  # the host never holds the whole buffer. Written atomically (temp + rename).
  rb$save <- function(path, chunk = 65536L) {
    header <- list(format = AG_RB_FORMAT, obs_dim = rb$obs_dim, act_dim = rb$act_dim,
                   capacity = rb$capacity, n_envs = rb$n_envs, rows = rb$layout$rows,
                   pos = rb$pos, size = rb$size, episode_start = rb$episode_start,
                   cur_start = rb$cur_start, prioritized = !is.null(rb$sampler),
                   alpha = rb$sampler$alpha, eps = rb$sampler$eps,
                   priorities = if (!is.null(rb$sampler)) rb$sampler$priorities())
    tmp <- tempfile(tmpdir = dirname(path), fileext = ".tmp")
    on.exit(unlink(tmp), add = TRUE)
    con <- file(tmp, "wb")
    ok <- FALSE
    tryCatch({
      serialize(header, con)
      if (rb$size > 0L) for (start in seq(0L, rb$size - 1L, by = chunk)) {
        k <- min(chunk, rb$size - start)
        writeBin(as.numeric(ag_buffer_read(rb$buf, start, k)), con, size = 4L)
      }
      ok <- TRUE
    }, finally = close(con))
    if (ok && !file.rename(tmp, path))
      stop("ag_replay_buffer: could not write '", path, "'", call. = FALSE)
    invisible(path)
  }

  class(rb) <- "ag_replay_buffer"
  rb
}

#' Load a replay buffer saved with \code{$save()}
#'
#' Restores into a new buffer on the current device. Also reads files written
#' by rltoolsR's replay buffer (\code{rltoolsR.replay_buffer.v1}), which are
#' not prioritized. As with any serialized R data, only read files you trust.
#'
#' @param path File written by \code{rb$save()}.
#' @param chunk Columns read per part.
#' @return An \code{\link{ag_replay_buffer}}.
#' @export
ag_replay_buffer_load <- function(path, chunk = 65536L) {
  con <- file(path, "rb")
  on.exit(close(con), add = TRUE)
  h <- tryCatch(unserialize(con), error = function(e) NULL)
  if (!is.list(h) || !(identical(h$format, AG_RB_FORMAT) ||
                       identical(h$format, AG_RB_FORMAT_RLTOOLS)))
    stop("'", path, "' is not a replay buffer file", call. = FALSE)
  pri <- isTRUE(h$prioritized)
  rb <- ag_replay_buffer(h$obs_dim, h$act_dim, h$capacity, h$n_envs, prioritized = pri,
                         alpha = if (pri) h$alpha else 0.6, eps = if (pri) h$eps else 1e-6)
  if (h$size > 0L) for (start in seq(0L, h$size - 1L, by = chunk)) {
    k <- min(chunk, h$size - start)
    v <- readBin(con, "double", n = h$rows * k, size = 4L)
    if (length(v) != h$rows * k) stop("'", path, "' is truncated", call. = FALSE)
    ag_buffer_write(rb$buf, matrix(v, h$rows, k), start)
  }
  rb$pos <- h$pos
  rb$size <- h$size
  rb$episode_start <- h$episode_start
  rb$cur_start <- h$cur_start
  if (pri) rb$sampler$load_priorities(h$priorities)
  rb
}

#' @export
print.ag_replay_buffer <- function(x, ...) {
  cat("<ag_replay_buffer> ", x$size, " / ", x$capacity, " transitions, obs ",
      x$obs_dim, ", act ", x$act_dim, ", ", x$n_envs, " env(s)",
      if (!is.null(x$sampler)) paste0(", prioritized (alpha ", x$sampler$alpha, ")"),
      "\n", sep = "")
  invisible(x)
}
